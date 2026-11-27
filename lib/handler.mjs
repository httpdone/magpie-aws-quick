// Serve one OpenAI Chat Completions request through an Amazon Quick chat turn.
//
// Quick has no OpenAI-shaped HTTP API: the desktop client talks to a bidi
// WebSocket (`session_connect` → `user_message` → text_delta/thinking_delta/…
// → message_complete). This module renders the OpenAI message list into Quick's
// text protocol, runs the turn on a cached thread, and re-emits the event
// stream as chat.completion / chat.completion.chunk.
//
// OpenAI `tools` have no wire channel either, but the model emits a
// caller-defined <tool_call>{json}</tool_call> convention as plain text when
// the instruction is inline and strong (verified live). lib/tools.mjs owns the
// preamble + streaming parser; results return as role:'tool' messages rendered
// into the next turn's text.
import { QuickSocket, QuickWsError } from './quickws.mjs'
import { OutpostError } from './outpost.mjs'
import { uploadImage } from './attach.mjs'
import { isPermissionReview, prepareReview, materializeReview, readReview } from './review.mjs'
import { normalizeTools, buildToolPreamble, createToolCallParser, createSourcesStripper } from './tools.mjs'
import { profileUsage } from './usage.mjs'
import {
  AGENT_MODES, MODELS, modeFor, effortFor, renderTranscript, renderDelta, renderToolCalls, textOf,
  byteLen, capText, setInstructions, UPSTREAM_TEXT_LIMIT, canonRole, pickImages,
} from './translate.mjs'
import {
  chainHashes, sessionScope, findContinuation, extendHash, replyMessage,
  replyAnchor, tailAnchor,
} from './sessions.mjs'

const SURFACE = 'QUICK_WORK_DESKTOP_WINDOWS'
const APP_VERSION = '1.2.4660'
// Reserve room for instructions while sizing the transcript and tools. The
// final payload gives instructions all remaining space in the encoded frame.
const SYSTEM_PROMPT_RESERVE = 32 * 1024
// Text room kept free when inline images claim the frame.
const IMAGE_TEXT_RESERVE = 8 * 1024
// Numeric plugin option with a default; 0 is a real value (disables the timer).
const msOpt = (v, fallback) => {
  if (v == null || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}
// Sentinel for "headerWaitMs elapsed while waiting for first output".
const HDR_WAIT = Symbol('header-wait')

export const PLAIN_TEXT_NOTE = 'Reply formatting: don\'t emit AWS Quick UI tags (such as <decision>, <option> or a sources block) — this client shows them as raw text; when the user needs to choose, ask with a short numbered list. This only affects how replies are formatted, not code or file contents.'
// With client tools, a formatting note alone reads as "put files in the chat":
// verified live, the model then pasted the code instead of calling Write.
// Placeholder tool name handed to the client when a <tool_call> can't be
// parsed (see consume): no client has it, so nothing runs.
export const INVALID_CALL_TOOL = 'aws_quick_invalid_tool_call'
export const TOOLS_STILL_APPLY = 'When the user asks you to make or change something (a page, script, file or project), do it on the user\'s machine with the client tools — write the file rather than pasting its contents into the reply.'

function errorResponse(status, message, type = 'api_error') {
  return new Response(JSON.stringify({ error: { message, type, code: status } }), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

// Quota exhaustion can surface as a NORMAL assistant reply — no error frame,
// just this fixed sentence as the entire message (first reported by
// kuk-888/aws-quick2api). Passed through, the client sees the model suddenly
// talking nonsense mid-task; detected, magpie fails over to another account.
const PSEUDO200_PHRASES = [
  "you've reached your usage limit. check your profile for details.",
  'you have reached your usage limit. check your profile for details.',
]

function normalizePseudo200(text) {
  return text.trimStart().toLowerCase().replace(/[‘’]/g, "'").replace(/\s+/g, ' ')
}

/** Exact whole-message match — the reply is nothing but the quota sentence. */
export function isPseudo200UsageLimitText(text) {
  const norm = normalizePseudo200(String(text ?? '')).trimEnd()
  if (!norm || norm.length > 140) return false
  return /^you(?:'ve| have) reached your usage limit\.\s*check your profile for details\.?$/i.test(norm)
}

/** Streaming prefix match — text so far could still become the quota sentence. */
export function isPseudo200UsageLimitPrefix(text) {
  const norm = normalizePseudo200(String(text ?? ''))
  if (!norm) return true
  if (norm.length > 80) return false
  return PSEUDO200_PHRASES.some((phrase) => phrase.startsWith(norm))
}

/** usage_limit failure type on an error/guardrail event payload (real error frames). */
function isUsageLimitPayload(p) {
  if (!p || typeof p !== 'object') return false
  for (const k of ['error_type', 'errorType', 'failure_type', 'failureType', 'code', 'reason', 'error', 'type']) {
    if (String(p[k] ?? '') === 'usage_limit') return true
  }
  return false
}

// ---------------------------------------------------------------------------
// Session-bucket failover: once a turn hits quota exhaustion, remember WHEN
// the account is usable again and fail fast (429) until then — magpie moves
// to the next account immediately instead of burning a turn per request.
// Keyed by identity; one refresh in flight per account.
// ---------------------------------------------------------------------------
const quotaWindows = new Map()
const quotaRefresh = new Map()

/** How long the account stays exhausted per profile/usage buckets. */
export function exhaustionMinutes(usage) {
  let mins = 0
  const s = usage?.session
  if (s?.exhausted) {
    const r = Number(s.resumeInMinutes)
    // server says when the window resumes; fall back to a conservative 30
    mins = Number.isFinite(r) && r > 0 ? r : 30
  }
  // the account is usable only when BOTH buckets have capacity — when the
  // monthly bucket is also dead the binding constraint is the later time,
  // not the session refill
  const m = usage?.monthly
  if (m?.exhausted) {
    const mm = m.resetsAt ? Math.max(1, (new Date(m.resetsAt).getTime() - Date.now()) / 60000) : 24 * 60
    mins = Math.max(mins, mm)
  }
  return mins
}

function quotaMarkExhausted(identity, getSession, log) {
  const win = quotaWindows.get(identity) || {}
  win.exhaustedAt = Date.now()
  if (!win.until || win.until < Date.now() + 60000) win.until = Date.now() + 15 * 60000 // conservative default until the probe refines it
  quotaWindows.set(identity, win)
  if (quotaRefresh.has(identity)) return
  const p = (async () => {
    try {
      const usage = await profileUsage(getSession)
      const mins = exhaustionMinutes(usage)
      const w = quotaWindows.get(identity)
      if (!w) return
      if (mins > 0) {
        w.until = Date.now() + mins * 60000
        w.probedAt = Date.now()
        log?.(`[quota] ${identity}: session/monthly bucket exhausted, skipping account for ~${Math.round(mins)}min`)
      } else if (usage && !usage.error && (usage.session || usage.monthly)) {
        // probe succeeded and buckets look fine — the limit we hit was
        // something else; keep only a short cooldown. A FAILED probe keeps
        // the default window instead of shortening it.
        w.until = Date.now() + 5 * 60000
        w.probedAt = Date.now()
        log?.(`[quota] ${identity}: usage_limit hit but buckets not exhausted; short 5min cooldown`)
      }
    } catch { /* keep the default window */ }
    finally { quotaRefresh.delete(identity) }
  })()
  quotaRefresh.set(identity, p)
}

/**
 * Minutes left if this identity is inside an exhaustion window, else 0.
 *
 * Re-probes profile/usage adaptively while the window runs (#19): the
 * failure-time probe is a single snapshot and session buckets refill
 * continuously, so a pure countdown either revives late or — when the
 * snapshot was wrong (e.g. monthly misread) — blocks for weeks. Probe
 * interval scales with the remaining window: min(30, max(3, left/10))
 * minutes, so short windows re-check fast and month-long ones cost at
 * most one cheap HTTP call per half hour, only when requests actually
 * arrive. Buckets showing capacity clear the window immediately.
 */
export async function quotaCooldownLeft(identity, getSession, log) {
  const win = quotaWindows.get(identity)
  if (!win?.until) return 0
  let left = win.until - Date.now()
  if (left <= 0) { quotaWindows.delete(identity); return 0 }
  if (!getSession) return left / 60000
  const intervalMs = Math.min(30, Math.max(3, (left / 60000) / 10)) * 60000
  const lastProbe = win.probedAt ?? win.exhaustedAt ?? 0
  if (Date.now() - lastProbe < intervalMs) return left / 60000
  win.probedAt = Date.now() // stamp first: concurrent requests share one probe
  if (!quotaRefresh.has(identity)) {
    const p = (async () => {
      try {
        const usage = await profileUsage(getSession)
        const w = quotaWindows.get(identity)
        if (!w) return
        // profileUsage never throws — a failed probe comes back as {error}
        // and must NOT read as recovery; require real bucket data for revival
        const probed = usage && !usage.error && (usage.session || usage.monthly)
        const mins = exhaustionMinutes(usage)
        if (probed && mins <= 0) {
          quotaWindows.delete(identity)
          log?.(`[quota] ${identity}: buckets have capacity again — account revived`)
        } else if (probed) {
          // still exhausted: correct the window with the fresh resume time
          w.until = Date.now() + mins * 60000
          w.probedAt = Date.now()
        }
      } catch { /* probe failed — the existing window stands */ }
      finally { quotaRefresh.delete(identity) }
    })()
    quotaRefresh.set(identity, p)
  }
  await quotaRefresh.get(identity)
  const w2 = quotaWindows.get(identity)
  if (!w2?.until) return 0
  left = w2.until - Date.now()
  if (left <= 0) { quotaWindows.delete(identity); return 0 }
  return left / 60000
}

/** Test hook: clear all quota windows. */
export function resetQuotaWindows() { quotaWindows.clear(); quotaRefresh.clear() }

/** Test hook: seed a window directly so probe logic can be exercised without waiting. */
export function seedQuotaWindowForTest(identity, win) { quotaWindows.set(identity, { ...win }) }

function errorType(err) {
  if (err?.kind === 'auth') return 'authentication_error'
  if (err?.status === 429 || err?.kind === 'quota') return 'rate_limit_error'
  return 'api_error'
}

function httpStatus(err) {
  if (err instanceof OutpostError || err instanceof QuickWsError) {
    if (err.kind === 'auth') return 401
    if (err.kind === 'abort') return 499
    if (err.kind === 'quota') return 429
    if (err.kind === 'spawn' || err.kind === 'timeout') return 503
  }
  return err?.status && err.status >= 400 && err.status < 600 ? err.status : 502
}

const sockets = new WeakMap() // Outpost → QuickSocket

async function socketFor(outpost, options) {
  let s = sockets.get(outpost)
  if (!s) {
    const outpostId = await outpost.outpostId()
    // OAuth-provisioned sessions (direct backend) have no local outpost; the
    // server accepts an empty outpostId. A *local* outpost returning none is a
    // real auth failure though.
    if (!outpostId && !outpost.direct) throw new OutpostError('outpost 未返回设备 id（/local/outpost-id），请确认已登录 Amazon Quick', { kind: 'auth', status: 401 })
    s = new QuickSocket({
      getSession: () => outpost.session(),
      outpostId,
      extra: {
        surfaceType: options?.surfaceType || SURFACE,
        appVersion: options?.appVersion || APP_VERSION,
        channel: options?.channel,
      },
      log: options?.log,
    })
    sockets.set(outpost, s)
  }
  return s
}

/** Usage object from a turn_usage event payload (snake_case wire fields). */
function usageFrom(u) {
  if (!u || typeof u !== 'object') return null
  const input = Number(u.input_tokens ?? u.inputTokens) || 0
  const cached = Number(u.cache_read_tokens ?? u.cached_input_tokens ?? u.cachedInputTokens) || 0
  const written = Number(u.cache_write_tokens ?? u.cacheWriteTokens) || 0
  const completion = Number(u.output_tokens ?? u.outputTokens) || 0
  // Quick reports non-cached input, cache reads and cache writes separately;
  // OpenAI prompt_tokens is the whole prompt side.
  const prompt = input + cached + written
  const total = Number(u.total_tokens ?? u.totalTokens) || prompt + completion
  const cost = Number(u.cost_usd ?? u.costUsd)
  const ctx = Number(u.context_window ?? u.contextWindow)
  const resolved = u.resolved_model_id ?? u.served_model_id ?? u.resolvedModelId
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total || prompt + completion,
    ...(cached ? { prompt_tokens_details: { cached_tokens: cached } } : {}),
    // Quick-reported extras; OpenAI clients ignore unknown usage fields.
    ...(Number.isFinite(cost) && cost > 0 ? { cost } : {}),
    ...(Number.isFinite(ctx) && ctx > 0 ? { context_window: ctx } : {}),
    ...(typeof resolved === 'string' && resolved ? { resolved_model_id: resolved } : {}),
  }
}

function addUsage(acc, u) {
  if (!u) return acc
  acc = acc ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  // Each agentic step re-reads the whole thread, so summing prompts counted
  // the same context once per step — clients size their context window from
  // prompt_tokens (Claude Code auto-compacted on it). The prompt is the
  // latest step's; only completions accumulate.
  if (u.prompt_tokens) {
    acc.prompt_tokens = u.prompt_tokens
    if (u.prompt_tokens_details?.cached_tokens) acc.prompt_tokens_details = { cached_tokens: u.prompt_tokens_details.cached_tokens }
    else delete acc.prompt_tokens_details
  }
  acc.completion_tokens += u.completion_tokens || 0
  acc.total_tokens = acc.prompt_tokens + acc.completion_tokens
  if (u.cost) acc.cost = (acc.cost || 0) + u.cost
  // Window and model are per-turn facts, not sums: keep the latest.
  if (u.context_window) acc.context_window = u.context_window
  if (u.resolved_model_id) acc.resolved_model_id = u.resolved_model_id
  return acc
}

const newCallId = () => `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`

/**
 * Normalized turn output:
 *   {type:'meta', threadId, reused, mode}
 *   {type:'text', text}                 visible answer delta
 *   {type:'reasoning', text}            thinking_delta
 *   {type:'tool_call', id, name, arguments}   parsed <tool_call> block (arguments = JSON string)
 *   {type:'usage', usage}               turn_usage (per message)
 *   {type:'note', text}                 non-fatal oddities worth showing
 *   {type:'done', …}                    terminal bookkeeping
 */
async function* runTurn({ outpost, socket, body, systemText, base, review, sessions, options, signal }) {
  const { agent_mode: mode, thinking_effort: effort } = base
  if (review) {
    // Reviews are one-shot even when Magpie supplied the main chat's session ID.
    try {
      const result = await readReview(socket, review, signal, msOpt(options?.idleTimeoutMs, 300_000))
      yield { type: 'meta', threadId: review.threadId, reused: false, mode }
      for (const u of result.usage) yield { type: 'usage', usage: usageFrom(u) }
      yield { type: 'text', text: result.text }
    } finally {
      // A primed review thread goes back to the pool for the next check.
      if (review.slot) review.slot.inflight = Math.max(0, review.slot.inflight - 1)
    }
    return
  }
  const messages = body.messages
  const convo = messages.filter((m) => m.role !== 'system' && m.role !== 'developer')
  const tcNone = body.tool_choice === 'none' || (typeof body.tool_choice === 'object' && body.tool_choice?.type === 'none')
  const tools = tcNone ? [] : normalizeTools(body.tools)
  // Quick's model sometimes calls a client tool through its own native
  // function calling (tool_start → tool_arguments → tool_result) instead of
  // our <tool_call> text convention; Quick has no such tool, so it errors
  // "Unknown tool" and the model rewrites everything as text — one observed
  // request took 285 s that way. consume() salvages those: a tool_arguments
  // event naming a client tool becomes a regular tool_call, then the turn is
  // stopped before the error+rewrite step. Names match exactly first, then
  // case-insensitively; the client's spelling is what gets emitted.
  const clientToolExact = new Set(tools.map((t) => t.name))
  const clientToolLower = new Map(tools.map((t) => [t.name.toLowerCase(), t.name]))
  const canonClientTool = (name) => {
    if (!clientToolExact.size || typeof name !== 'string' || !name) return null
    if (clientToolExact.has(name)) return name
    return clientToolLower.get(name.toLowerCase()) ?? null
  }
  const systemReserve = Math.min(byteLen(systemText), SYSTEM_PROMPT_RESERVE)
  const stats = {}
  const sessionKey = typeof options?.sessionKey === 'string' ? options.sessionKey.slice(0, 128) : ''
  // Quick's own tool_start/tool_progress/tool_result events are agent-internal;
  // showQuickTools surfaces each tool_start as a [Quick: name] reasoning line.
  const showQuickTools = options?.showQuickTools !== false
  // custom_prompt is a real per-turn instruction channel (verified live). Use
  // it to tell the model this session is app-driven: with client tools, steer
  // local-environment work to the <tool_call> protocol; on direct backends
  // (OAuth / Credential Manager — no outpost, hence no connected device) also
  // say built-in device tools cannot run, so the model doesn't burn agentic
  // turns on guaranteed failures. The local-outpost path may genuinely have a
  // device attached, so the claim stays off there.
  const directBackend = !!outpost?.direct
  const baseCustomPrompt = tools.length
    ? directBackend
      ? 'Calling context: this chat is driven by an external application. The client tools listed in the message are executed by that application on the user\'s machine; this session has no connected device, so built-in device tools (local files, apps) cannot run here. For questions about the user\'s project, files or code, call the client tools to read current contents — earlier tool results may be partial, and a fresh read beats guessing.'
      : 'Calling context: this chat is driven by an external application. The client tools listed in the message are executed by that application on the user\'s machine — for questions about the user\'s project, files or code, call them to read current contents; earlier tool results may be partial, and a fresh read beats guessing.'
    : directBackend
      ? 'Calling context: this chat is driven by an external application and has no connected device. Built-in tools that need the user\'s device (local files, apps) cannot run in this session — say so plainly rather than attempting them.'
      : ''
  // Quick's own system prompt asks for Quick-app UI markup (decision cards,
  // artifacts). External clients render plain text/Markdown, so that markup
  // leaks as raw tags. Scoped so code/file output (e.g. HTML <option>) is unaffected.
  const customPrompt = [baseCustomPrompt, tools.length ? TOOLS_STILL_APPLY : '', PLAIN_TEXT_NOTE].filter(Boolean).join(' ')
  const textCap = Math.max(8 * 1024, UPSTREAM_TEXT_LIMIT - systemReserve - byteLen(customPrompt) - 1024)

  // Thread identity: account, agent mode and the caller's system text (which
  // separates main agent, subagents and the permission reviewer). Thinking
  // effort is a per-turn field, and tools are tracked on the entry instead —
  // Claude Code adds tools mid-session (ToolSearch, skills), and each change
  // used to throw away a thread holding the whole conversation (#11).
  const identity = options?.identity || 'local'
  const scope = sessionScope(identity, mode, systemText, null)
  const hashes = chainHashes(scope, convo)
  const cont = sessions ? findContinuation(sessions, hashes, convo) : undefined
  // Entries taken for reuse. Matching removes them up front so two requests
  // can't append to one thread; if this turn then fails or is cancelled
  // without committing a new entry, they go back — otherwise one 499/502/503
  // would strand a thread holding the whole conversation.
  const taken = cont ? [[cont.key, cont.entry]] : []
  // Fuzzy resume: context extensions (pi-magic-context, rewind hooks, …) can
  // prune or rewrite old messages, breaking the strict prefix chain. Anchor on
  // our last reply echo + its 3 predecessors — content hashes survive the
  // position shifts such edits cause. Only the LAST assistant can yield an
  // appendable delta anyway.
  let fuzzy
  if (!cont && sessions) {
    let a = -1
    for (let i = convo.length - 1; i >= 0; i--) {
      if (canonRole(convo[i]?.role) === 'assistant') { a = i; break }
    }
    if (a >= 3) {
      const rk = replyAnchor(scope, convo[a])
      const tk = tailAnchor(scope, convo.slice(a - 3, a))
      const hit = sessions.findEntry((e) => e.replyKey === rk && e.tailKey === tk)
      if (hit) {
        const delta = convo.slice(a + 1)
        if (delta.length && delta.every((m) => ['user', 'tool', 'function'].includes(canonRole(m.role)))) {
          sessions.take(hit.key)
          taken.push([hit.key, hit.entry])
          fuzzy = { entry: hit.entry, delta }
        }
      }
    }
  }
  // Session-keyed fallback: gateways like magpie stamp every request of a
  // chat with a stable session id (x-magpie-session). When content matching
  // fails entirely — heavy client-side compaction rewrites even our reply
  // echo — the session boundary still identifies the conversation, so trust
  // it and append whatever follows the last assistant message. Verified
  // matches (strict/fuzzy) always win; this is the last resort.
  let sessHit
  if (!cont && !fuzzy && sessions && sessionKey) {
    const sk = `sess:${scope}:${sessionKey}`
    if (sessions.has(sk)) {
      let a = -1
      for (let i = convo.length - 1; i >= 0; i--) {
        if (canonRole(convo[i]?.role) === 'assistant') { a = i; break }
      }
      if (a >= 0) {
        const delta = convo.slice(a + 1)
        if (delta.length && delta.every((m) => ['user', 'tool', 'function'].includes(canonRole(m.role)))) {
          const entry = sessions.take(sk)
          // One session id covers every subagent of a Claude Code session
          // (and a gateway may move each request to another account), so the
          // id alone can splice one subagent's tool results into another
          // subagent's thread. Tool results must answer calls this thread
          // made; a plain user follow-up still resumes.
          const resultIds = delta.filter((m) => canonRole(m.role) === 'tool').map((m) => m.tool_call_id).filter(Boolean)
          const ours = new Set(entry?.callIds ?? [])
          if (entry && resultIds.every((id) => ours.has(id))) {
            taken.push([sk, entry])
            sessHit = { entry, delta }
          } else if (entry) sessions.set(sk, entry)
        }
      }
    }
  }
  const reuse = cont ?? fuzzy ?? sessHit
  // A tool preamble that would leave under ~8 KB for the actual message
  // degrades smoothly: full JSON schema → compact typed signatures (keeps all
  // parameter names, required !, types, enums) → lite → namesOnly.
  const fitPreamble = (list = tools) => {
    let pre = buildToolPreamble(list, body.tool_choice, { stats })
    if (byteLen(pre) > textCap - 8192) {
      const compact = buildToolPreamble(list, body.tool_choice, { compact: true })
      if (byteLen(compact) < byteLen(pre)) { pre = compact; stats.truncated = true }
    }
    if (byteLen(pre) > textCap - 6144) {
      const lite = buildToolPreamble(list, body.tool_choice, { lite: true })
      if (byteLen(lite) < byteLen(pre)) { pre = lite; stats.truncated = true }
    }
    if (byteLen(pre) > textCap - 4096) {
      const names = buildToolPreamble(list, body.tool_choice, { namesOnly: true })
      if (byteLen(names) < byteLen(pre)) { pre = names; stats.truncated = true }
    }
    return pre
  }
  // A reused thread already holds the spec of every tool described in it
  // (entry.toolNames), and whatever rides in `text` stays in Quick's
  // history — so later turns get a short reminder, plus the full spec of
  // just the tools added since. Fresh threads, including the stale-thread
  // retry below, get the full spec. An entry reloaded from disk without
  // toolNames gets everything once.
  //
  // Format decay: the reminder alone is not enough to keep smaller models
  // emitting <tool_call> blocks — Quick's own system prompt dominates the
  // context and the spec scrolls away as the thread grows, so after a
  // handful of reminder-only turns the model starts answering in plain
  // text (the "no tools" failure agents surface). Re-anchor periodically:
  // every `toolSpecReanchorTurns` reminder turns the full spec rides again.
  // Cheap insurance — Quick's server-side prompt cache absorbs most of the
  // repeated bytes.
  const reanchorAt = Math.max(0, Math.floor(msOpt(options?.toolSpecReanchorTurns, 6)))
  const priorSpecTurns = Number.isFinite(reuse?.entry?.specTurns) ? reuse.entry.specTurns : reanchorAt
  const reanchor = !!(reuse && tools.length && reanchorAt > 0 && priorSpecTurns + 1 >= reanchorAt)
  const known = reuse?.entry?.toolNames ? new Set(reuse.entry.toolNames) : null
  const added = reuse ? tools.filter((t) => !known?.has(t.name)) : []
  const preamble = !tools.length ? ''
    : !reuse ? fitPreamble()
      : reanchor ? fitPreamble()
        : [added.length ? fitPreamble(added) : '', buildToolPreamble(tools, body.tool_choice, { reminder: true })].filter(Boolean).join('\n\n')
  // what this thread will have seen once this turn lands
  let toolNames = [...new Set([...(reuse ? known ?? [] : []), ...tools.map((t) => t.name)])]
  // Turns since the full spec last rode this thread. Zero when this turn
  // (re-)anchors; the fresh-thread retry below resets it too. Entries from
  // before this counter existed are treated as overdue (reanchor at once).
  let specTurns = (reanchor || !reuse) ? 0 : priorSpecTurns + 1
  const textBudget = Math.max(4 * 1024, textCap - byteLen(preamble))

  // P2 — argument validation feedback. With lite/names-only preambles the
  // model guesses parameters, and schemas its guess violates stay invisible
  // to it: the client fails the call, the thread history shows a call that
  // "ran", and the model repeats the same shape. Validate each parsed call
  // against the slimmed schema; misses ride the next turn as corrections.
  // Deliberately shallow — required keys and top-level primitive types only;
  // false positives are worse than misses here, the client remains the
  // authority on what actually runs.
  const toolSchemas = new Map(tools.map((t) => [t.name, t.parameters]))
  const TYPECHECKS = {
    string: (v) => typeof v === 'string',
    number: (v) => typeof v === 'number',
    integer: (v) => typeof v === 'number',
    boolean: (v) => typeof v === 'boolean',
    array: (v) => Array.isArray(v),
    object: (v) => v !== null && typeof v === 'object' && !Array.isArray(v),
  }
  const validateCallArguments = (name, args) => {
    const schema = toolSchemas.get(name)
    const props = schema?.properties
    if (!props || typeof props !== 'object') return []
    const problems = []
    const a = args && typeof args === 'object' ? args : {}
    for (const key of Array.isArray(schema.required) ? schema.required : []) {
      if (!(key in a) || a[key] == null) problems.push(`missing required "${key}"`)
    }
    for (const [key, val] of Object.entries(a)) {
      const spec = props[key]
      const type = spec?.type
      if (typeof type === 'string' && TYPECHECKS[type] && val != null && !TYPECHECKS[type](val)) {
        problems.push(`"${key}" should be ${type}, got ${Array.isArray(val) ? 'array' : typeof val}`)
      }
    }
    return problems.slice(0, 4)
  }
  // This turn's corrections, persisted on the session entry for the next one.
  const argErrors = []

  // Images ride user_message.attachments: small ones inline as base64 inside
  // the frame; larger ones go through Quick's cloud attachment upload
  // (lib/attach.mjs) on every backend — the outpost session's sncs-session
  // carries the id_token/ws_host the upload API needs (verified live with a
  // local-outpost sign-in). Only this turn's images are sent; inline bytes
  // shrink the text budget, uploaded ones don't (they carry no base64 on the
  // wire).
  const lastAsst = convo.findLastIndex((m) => canonRole(m.role) === 'assistant')
  const imageSlice = reuse ? reuse.delta : convo.slice(lastAsst + 1)
  const images = pickImages(imageSlice, Math.max(0, textBudget - IMAGE_TEXT_RESERVE), { upload: true })
  const turnBudget = textBudget - images.bytes
  const frameCap = textCap - images.bytes

  let threadId = reuse ? reuse.entry.threadId : crypto.randomUUID()
  const imageNoteFor = (uploadFailed) => {
    const n = images.skipped + uploadFailed
    return n ? `\n\n[${n} image(s) not delivered: too large, upload failed, or not inline data]` : ''
  }
  // Deliver the queued uploads into tid's cloud workspace, in parallel.
  // Failures degrade to the not-delivered note; aborts propagate.
  const uploadAll = async (tid) => {
    if (!images.toUpload.length) return { entries: [], failed: 0 }
    const sess = await outpost.session()
    const devId = await Promise.resolve(outpost.outpostId?.()).catch(() => '')
    const results = await Promise.allSettled(images.toUpload.map((img) =>
      uploadImage({
        session: sess, image: img, name: img.name, threadId: tid, outpostId: devId || undefined,
        fetchImpl: options?.fetchImpl ?? globalThis.fetch, signal,
      })
    ))
    const entries = []
    let failed = 0
    for (const r of results) {
      if (r.status === 'fulfilled') { entries.push(r.value); continue }
      if (signal?.aborted || r.reason?.name === 'AbortError') throw r.reason
      failed++
      options?.log?.(`[aws-quick] image upload failed: ${String(r.reason?.message ?? r.reason).slice(0, 200)}`)
    }
    return { entries, failed }
  }
  const uploaded = await uploadAll(threadId)
  const imageNote = imageNoteFor(uploaded.failed)
  const attachments = images.attachments.concat(uploaded.entries)

  let text
  let reused = false
  if (reuse) {
    // A turn that ended in native-tool salvage was stopped server-side: the
    // reused thread's history holds an empty assistant message + "You stopped
    // the response." — NOT the call the client is now answering. Re-state it
    // (same rendering transcript history uses for tool_calls) ahead of the
    // delta, once: the entry written after this turn carries no flag.
    const salvagePrefix = reuse.entry.salvageCalls?.length
      ? `[Your previous turn was interrupted after you called these tools; the client executed them. Ignore any "Unknown tool" error recorded for them: they run on the client, so keep calling client tools with <tool_call> blocks.]\nassistant: ${renderToolCalls(reuse.entry.salvageCalls)}\n\n`
      : ''
    const dc = reuse.entry.droppedCall
    const droppedPrefix = dc
      ? `[Note: your previous <tool_call>${dc.name ? ` for ${dc.name}` : ''} ${dc.reason === 'unterminated'
        ? 'was cut off before </tool_call>, so the client never ran it. Call it again; if the content is long, split it into several smaller calls (e.g. write the file in parts).'
        : 'was not valid JSON, so the client never ran it. Call it again with valid JSON (escape newlines and quotes inside strings).'
      } Any error for ${INVALID_CALL_TOOL} below is just the placeholder for that failed call.]\n\n`
      : ''
    const argErrPrefix = reuse.entry.argErrors?.length
      ? `[Corrections — your previous tool call(s) had invalid arguments and failed on the client: ${reuse.entry.argErrors.join('; ')}. Fix the arguments and call again.]

`
      : ''
    const prefix = argErrPrefix + salvagePrefix + droppedPrefix
    text = prefix + renderDelta(reuse.delta, { maxBytes: turnBudget - byteLen(prefix), stats }) + imageNote
    reused = true
  } else {
    text = renderTranscript(convo, '', { maxBytes: turnBudget, stats }) + imageNote
  }
  // The tool convention rides inside the user_message text (there is no
  // client-tool channel); on reused threads only the short reminder rides,
  // since every byte of text stays in the thread and counts toward context.
  let upstreamText = preamble ? `${preamble}\n\n${text}` : text
  if (byteLen(upstreamText) > frameCap) {
    // Last resort before the tail-cutting hard cap: shed the preamble so the
    // user's message survives intact (tool calling degrades, the answer doesn't).
    upstreamText = byteLen(text) <= frameCap ? text : capText(text, frameCap)
    stats.truncated = true
  }
  if (!upstreamText.trim()) upstreamText = '(empty)'

  const state = { text: '', reasoning: '', usage: null, calls: [] }
  // Client-tool calls salvaged from Quick's native channel this turn
  // ({name, arguments} — arguments already JSON). Quick discards the stopped
  // step server-side, so the session entry carries these and the NEXT reused
  // turn re-states them before the tool results (see the reuse branch below).
  const salvageCalls = []
  const parser = tools.length ? createToolCallParser() : null
  // Quick's memory citations (<sources>[…](mem://…)</sources>) still leak
  // past the prompt note now and then; strip them from the visible text.
  const sources = createSourcesStripper()
  // Set when a <tool_call> block had to be dropped: the reused thread's
  // history shows the call as made, so the next turn says it never ran.
  let droppedCall = null
  let fedText = false
  let sawOutput = false

  const payload = { ...base, text: upstreamText }
  if (tools.length) payload.web_search_enabled = false
  if (attachments.length) payload.attachments = attachments
  const clippedSystem = setInstructions(payload, systemText, customPrompt, threadId)

  function emitParsed(r) {
    // r: {text, calls:[{name,arguments}]} — yields normalized items, mutates state.
    const out = []
    const text = r.text ? sources.push(r.text) : ''
    if (text) { state.text += text; out.push({ type: 'text', text }) }
    for (const c of r.calls) {
      const id = newCallId()
      state.calls.push({ id, name: c.name })
      const problems = validateCallArguments(c.name, c.arguments)
      if (problems.length) {
        argErrors.push(`${c.name}: ${problems.join(', ')}`)
        out.push({ type: 'note', text: `[quick] ${c.name} 的参数不符合 schema(${problems.join(', ')}),已在下一轮提示模型纠正` })
      }
      out.push({ type: 'tool_call', id, name: c.name, arguments: JSON.stringify(c.arguments ?? {}) })
    }
    if (out.length) sawOutput = true
    return out
  }

  async function* consume(turnIt, tid) {
    // Salvage state for native client-tool calls (see canonClientTool above):
    // tool_ids that produced a call; salvageCalls (runTurn scope) holds them.
    const salvagedIds = new Set()
    // native client calls started but whose arguments haven't arrived yet —
    // a step can hold several (parallel Edits); stopping on the first
    // tool_result would cut the rest off
    const pendingNative = new Set()
    // End the Quick turn once salvaged calls exist and the model step ends —
    // stopTurn tells Quick to stop (its own stop_generation frame) and the
    // turn signal abort wakes the parked socket.turn waiter so closing the
    // generator can't hang on a silent Quick.
    const endSalvage = () => { stopTurn(tid); turnCtl.abort() }
    // Pseudo-200 quota probe: hold back the turn's first text while it could
    // still be the fixed "usage limit" sentence (see isPseudo200* above). The
    // buffer flushes as ordinary text the moment it diverges; if the turn
    // ends with nothing but that sentence it is quota exhaustion, not a reply.
    let quotaProbe = true
    let quotaBuf = ''
    for await (const evt of turnIt) {
      if (progressEvent(evt)) bumpWatchdog()
      const t = evt.type
      const p = evt.payload || {}
      if (t === 'text_delta' && typeof p.delta === 'string') {
        fedText = true
        let delta = p.delta
        if (quotaProbe) {
          quotaBuf += delta
          if (quotaBuf.length <= 160 && isPseudo200UsageLimitPrefix(quotaBuf)) continue
          quotaProbe = false
          delta = quotaBuf
          quotaBuf = ''
        }
        const r = parser ? parser.push(delta) : { text: delta, calls: [] }
        yield* emitParsed(r)
      } else if (t === 'thinking_delta' && typeof p.delta === 'string') {
        sawOutput = true
        state.reasoning += p.delta
        yield { type: 'reasoning', text: p.delta }
      } else if (t === 'tool_start' && showQuickTools && typeof p.tool_name === 'string' && p.tool_name && !canonClientTool(p.tool_name)) {
        // Quick runs its own tools server-side; surface each one as a
        // reasoning line so long agentic turns don't look dead. Client-tool
        // names are skipped — the salvaged call itself is the surface.
        const note = `\n[Quick: ${p.tool_name}]\n`
        sawOutput = true
        state.reasoning += note
        yield { type: 'reasoning', text: note }
      } else if (t === 'tool_start' && p.tool_id && canonClientTool(p.tool_name)) {
        pendingNative.add(p.tool_id)
      } else if (t === 'tool_arguments') {
        if (p.tool_id) pendingNative.delete(p.tool_id)
        // Native client call: the full args object just arrived. Emit it like
        // a parsed <tool_call>; the step-end stop happens on the next
        // turn_usage (yielded first) or a tool_result for a salvaged id.
        const canon = canonClientTool(p.tool_name)
        const dup = p.tool_id && salvagedIds.has(p.tool_id)
        if (canon && !dup && p.arguments && typeof p.arguments === 'object') {
          if (p.tool_id) salvagedIds.add(p.tool_id)
          const id = newCallId()
          state.calls.push({ id, name: canon })
          salvageCalls.push({ name: canon, arguments: JSON.stringify(p.arguments) })
          sawOutput = true
          yield { type: 'tool_call', id, name: canon, arguments: JSON.stringify(p.arguments) }
        }
      } else if (t === 'tool_result' && salvagedIds.has(p.tool_id)) {
        // Quick finished (or failed) a salvaged call — stop before the
        // model's rewrite step burns another agent turn, unless sibling
        // calls of the same step are still streaming their arguments
        // (turn_usage ends the step either way).
        if (pendingNative.size) continue
        endSalvage()
        return
      } else if (t === 'turn_usage') {
        const u = usageFrom(p)
        if (u) { state.usage = addUsage(state.usage, u); yield { type: 'usage', usage: u } }
        if (salvageCalls.length) { endSalvage(); return }
      } else if (t === 'message_complete') {
        // Fallback: deltas can be dropped; the complete payload carries full text.
        // Only when NO delta arrived — parser buffer would double-count otherwise.
        if (!fedText && typeof p.text === 'string' && p.text) {
          // Whole message in one frame: the quota sentence is detectable outright.
          if (isPseudo200UsageLimitText(p.text)) {
            throw new QuickWsError('Quick 账号用量已用尽（usage limit），请等待额度重置或更换账号', { kind: 'quota' })
          }
          const r = parser ? parser.push(p.text) : { text: p.text, calls: [] }
          yield* emitParsed(r)
        }
      } else if (t === 'message_start' || t === 'connected' || t === 'session_established' || t === 'keepalive') {
        // noise
      } else if (t === 'status') {
        const s = p.status || p.message
        // unknown_message_type {received_type:'session_init'}: a server-side
        // status about a frame we never send, seen on every connect with or
        // without an outpostId — noise, not something the user can act on.
        if (s && !/pong|ok|unknown_message_type/i.test(String(s))) yield { type: 'note', text: `[quick] ${s}` }
      } else if (t === 'guardrail_blocked' || t === 'error') {
        // Error details may live in payload OR at the event's top level;
        // dump the whole event (truncated) when nothing readable is found.
        const detail = p.message || p.reason || p.error || evt.message || evt.reason || evt.error || evt.code
        const extra = detail ? '' : ` ${JSON.stringify(evt).slice(0, 400)}`
        if (isUsageLimitPayload(p) || isUsageLimitPayload(evt)) {
          throw new QuickWsError('Quick 账号用量已用尽（usage limit），请等待额度重置或更换账号', { kind: 'quota' })
        }
        throw new QuickWsError(`${detail || `quick ${t}`}${extra}`, { kind: 'stream' })
      }
      // tool_progress, runs_updated, task_list_update, title_generated,
      // session_sync … are agent-internal; Quick executes its own tools, so
      // they don't map to OpenAI tool_calls (tool_start is surfaced above as
      // a reasoning note; tool_arguments/tool_result drive the client-tool
      // salvage above).
    }
    if (quotaProbe && quotaBuf) {
      // The turn ended with every text delta still consistent with the quota
      // sentence. An exact match is exhaustion — fail as 429 so magpie moves
      // to the next account; anything else was a real reply that merely
      // started alike (or a truncated one): flush it through.
      if (isPseudo200UsageLimitText(quotaBuf)) {
        throw new QuickWsError('Quick 账号用量已用尽（usage limit），请等待额度重置或更换账号', { kind: 'quota' })
      }
      const r = parser ? parser.push(quotaBuf) : { text: quotaBuf, calls: [] }
      yield* emitParsed(r)
      quotaBuf = ''
    }
    if (parser) {
      const tail = parser.end()
      if (tail.text) yield* emitParsed({ text: tail.text, calls: [] })
      for (const c of tail.calls ?? []) {
        const id = newCallId()
        state.calls.push({ id, name: c.name })
        yield { type: 'tool_call', id, name: c.name, arguments: JSON.stringify(c.arguments ?? {}) }
      }
      if (tail.dropped) {
        droppedCall = { reason: tail.dropped, name: tail.droppedName || '' }
        yield {
          type: 'note',
          text: tail.dropped === 'unterminated'
            ? '[quick] 模型的工具调用没写完（回复在 </tool_call> 之前就结束了），这次调用没有执行，已让模型重发'
            : '[quick] 模型的工具调用格式有误（JSON 无法解析），这次调用没有执行，已让模型重发',
        }
        // A turn that ends with no tool call makes agent clients stop and
        // wait for the user. Hand them a placeholder call to a tool that
        // doesn't exist instead: nothing runs, the client answers with an
        // error result and keeps its loop going (verified with Claude Code),
        // and the next turn tells the model what to resend.
        if (!state.calls.length) {
          const id = newCallId()
          const args = JSON.stringify({ tool: droppedCall.name, reason: tail.dropped })
          state.calls.push({ id, name: INVALID_CALL_TOOL })
          yield { type: 'tool_call', id, name: INVALID_CALL_TOOL, arguments: args }
        }
      }
    }
    const rest = sources.end()
    if (rest) { state.text += rest; sawOutput = true; yield { type: 'text', text: rest } }
  }

  yield { type: 'meta', threadId, reused, mode }
  if (stats.truncated || clippedSystem) yield { type: 'note', text: '上下文超过 Quick 64KB 上限，已截断较早内容' }

  // Idle watchdog: every Quick event is progress except transport noise
  // (keepalive/ping/pong and ok-status pings). The timer aborts an internal
  // controller linked to the request signal, so socket.turn wakes even when
  // Quick sends nothing at all; watchdogFired distinguishes it from a client
  // abort when translating the error below.
  const idleMs = msOpt(options?.idleTimeoutMs, 300_000)
  const turnCtl = new AbortController()
  const linkRequestAbort = () => turnCtl.abort()
  if (signal) {
    if (signal.aborted) turnCtl.abort()
    else signal.addEventListener('abort', linkRequestAbort, { once: true })
  }
  let watchdogFired = false
  let watchdogTimer = null
  const bumpWatchdog = () => {
    if (!idleMs) return
    if (watchdogTimer) clearTimeout(watchdogTimer)
    watchdogTimer = setTimeout(() => { watchdogFired = true; turnCtl.abort() }, idleMs)
    watchdogTimer.unref?.()
  }
  const progressEvent = (evt) => {
    const t = evt?.type
    if (t === 'keepalive' || t === 'ping' || t === 'pong') return false
    // UI bookkeeping, not generation: a stuck turn keeps emitting these.
    if (t === 'runs_updated' || t === 'task_list_update' || t === 'title_generated' || t === 'session_summary_updated' || t === 'session_sync') return false
    if (t === 'status') {
      const s = evt?.payload?.status || evt?.payload?.message || ''
      if (/pong|ok/i.test(String(s))) return false
    }
    return true
  }
  // Quick's own stop message (the frame the desktop sends on stop): socket.stop
  // sends it AND arms the per-thread settle window, so the next turn() on this
  // thread waits out Quick's stale terminal-event tail instead of losing its
  // user_message. Covers watchdog timeout, client abort and tool salvage.
  const stopTurn = (tid) => {
    try { socket.stop(tid) } catch {}
  }
  // Translate a watchdog/client abort into the error the caller should see;
  // returns null for ordinary turn errors (which may still hit the retry).
  const turnTermination = (e, tid) => {
    if (watchdogFired) {
      stopTurn(tid)
      return new QuickWsError(`Quick 已 ${idleMs / 1000}s 没有进展，已中止本轮 (idleTimeoutMs)`, { kind: 'timeout' })
    }
    if (signal?.aborted || e?.kind === 'abort') {
      stopTurn(tid)
      return e
    }
    return null
  }
  bumpWatchdog()

  const produced = () => state.text.length > 0 || state.calls.length > 0 || state.reasoning.length > 0
  const restoreTaken = () => {
    for (const [k, e] of taken) if (!sessions.has(k)) sessions.set(k, e)
    taken.length = 0
  }
  let turnOk = false
  try {
    for await (const o of consume(socket.turn(threadId, payload, { signal: turnCtl.signal }), threadId)) yield o
    turnOk = true
  } catch (e) {
    const term = turnTermination(e, threadId)
    if (term) throw term
    if (!(reuse && sessions) || produced() || e?.kind === 'auth' || e?.kind === 'timeout') throw e
    // Cached thread may be gone server-side: retry once on a fresh thread with
    // the full transcript (same policy as magpie-comate). Uploaded files live
    // in the dead thread's workspace, so they are delivered again for `fresh`.
    const fresh = crypto.randomUUID()
    taken.length = 0 // the cached thread is the dead one — nothing to put back
    toolNames = tools.map((t) => t.name)
    specTurns = 0
    const up2 = await uploadAll(fresh)
    const retryAttachments = images.attachments.concat(up2.entries)
    const full = renderTranscript(convo, '', { maxBytes: turnBudget, stats }) + imageNoteFor(up2.failed)
    const fullPreamble = tools.length ? fitPreamble() : ''
    const fullText = fullPreamble ? `${fullPreamble}\n\n${full}` : full
    const capped = byteLen(fullText) > frameCap ? capText(fullText, frameCap) : fullText
    state.text = ''
    state.reasoning = ''
    state.usage = null
    state.calls = []
    salvageCalls.length = 0
    droppedCall = null
    fedText = false
    sawOutput = false
    threadId = fresh
    reused = false
    yield { type: 'meta', threadId: fresh, reused: false, mode }
    payload.text = capped.trim() || '(empty)'
    payload.message_id = crypto.randomUUID()
    if (retryAttachments.length) payload.attachments = retryAttachments
    else delete payload.attachments
    setInstructions(payload, systemText, customPrompt, fresh)
    try {
      for await (const o of consume(socket.turn(fresh, payload, { signal: turnCtl.signal }), fresh)) yield o
      turnOk = true
    } catch (e2) {
      const term2 = turnTermination(e2, fresh)
      if (term2) throw term2
      throw e2
    }
  } finally {
    if (watchdogTimer) clearTimeout(watchdogTimer)
    signal?.removeEventListener?.('abort', linkRequestAbort)
    // errors, watchdog and client cancels (generator return) all land here
    if (!turnOk && sessions) restoreTaken()
  }

  // Remember where this conversation stands for the next request.
  if (sessions && (state.text || state.calls.length)) {
    const reply = replyMessage(state.text, state.calls)
    const key = extendHash(hashes[convo.length], reply)
    // salvageCalls ride the entry so the next reused turn can re-state the
    // interrupted call (memory only — the file store keeps hashes + threadIds,
    // never message content, so a reloaded entry simply lacks the prefix).
    const salvaged = { ...(salvageCalls.length ? { salvageCalls } : {}), ...(droppedCall ? { droppedCall } : {}) }
    // Reactive re-anchor: the client answered tool calls (the delta carried
    // tool results) yet the model replied without making any new call —
    // the plain-text failure this adapter's reminder-only turns drift into.
    // Force the next turn to re-anchor regardless of the periodic counter.
    const answeredWork = reuse?.delta?.some((m) => canonRole(m.role) === 'tool')
    const drifted = !!(answeredWork && !state.calls.length)
    const nextSpecTurns = drifted ? reanchorAt : specTurns
    const corrective = { ...salvaged, specTurns: nextSpecTurns, ...(argErrors.length ? { argErrors: argErrors.slice(0, 3) } : {}) }
    sessions.set(key, {
      threadId,
      toolNames,
      replyKey: replyAnchor(scope, reply),
      tailKey: tailAnchor(scope, convo.slice(-3)),
      ...corrective,
    })
    // Session-keyed entry self-perpetuates: rewritten every turn so the
    // fallback survives however much the client rewrites its history.
    if (sessionKey) sessions.set(`sess:${scope}:${sessionKey}`, { threadId, toolNames, callIds: state.calls.map((c) => c.id), ...corrective })
  } else if (sessions) restoreTaken()
  yield { type: 'done', threadId, text: state.text, reasoning: state.reasoning, usage: state.usage }
}

function makeId(threadId) {
  return `chatcmpl-quick-${threadId?.slice(0, 8) ?? 'x'}-${Date.now().toString(36)}`
}

/**
 * @param {object} p
 * @param {object} p.body       parsed Chat Completions request
 * @param {object} p.outpostMgr createOutpostManager() result
 * @param {object} [p.sessions] session store (lib/sessions.mjs)
 * @param {object} [p.options]  plugin options
 * @param {string} [p.effortHeader] X-Magpie-Effort style header value
 * @param {AbortSignal} [p.signal]
 */
async function runChatCompletions({ body, outpostMgr, sessions, options, effortHeader, signal }) {
  if (!body || typeof body !== 'object') return errorResponse(400, 'request body must be JSON', 'invalid_request_error')
  if (!Array.isArray(body.messages) || !body.messages.length) {
    return errorResponse(400, 'messages is required', 'invalid_request_error')
  }
  if (body.messages.some(m => !m || typeof m !== 'object')) {
    return errorResponse(400, 'each message must be an object', 'invalid_request_error')
  }
  const model = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : 'smart'
  const messages = body.messages
  const systemText = messages.filter(m => m.role === 'system' || m.role === 'developer').map(m => textOf(m.content)).join('\n\n')
  const mode0 = modeFor(model)
  // advanced plan-gating (#23): once the reactive probe below proved this
  // plan lacks advanced, map to smart from the start so thread scope stays
  // consistent and no turn is burned on the doomed attempt.
  const mode = mode0 === 'advanced' && advancedDowngrades.has(options?.identity || 'local') ? 'smart' : mode0
  const effort = effortFor(effortHeader, body)
  const base = {
    message_id: crypto.randomUUID(), agent_mode: mode,
    user_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
    user_locale: (Intl.DateTimeFormat().resolvedOptions().locale || 'en-US').replace('_', '-'),
  }
  if (effort) base.thinking_effort = effort
  if (options?.chatAgentId) base.chat_agent_id = options.chatAgentId
  let review
  if (isPermissionReview(systemText, messages, body.tools)) {
    const threadId = crypto.randomUUID()
    try {
      review = prepareReview(messages, systemText, base, threadId)
    } catch (err) {
      return errorResponse(err.status ?? 422, err.message, 'invalid_request_error')
    }
  }
  const identity = options?.identity || 'local'
  let outpost
  try {
    outpost = await outpostMgr.ensure()
  } catch (e) {
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }
  const cdMin = await quotaCooldownLeft(identity, () => outpost.session(), options?.log)
  if (cdMin > 0) {
    // Wording matters beyond readability: magpie's router parses the reset
    // time out of a refusal's text to decide how long the account rests —
    // its resetsInWords reads "resets in 3h 32m" exactly, while "resumes in
    // ~N minutes" parsed as nothing and the account fell back to the blind
    // 15-minute backoff ladder (issue #19's never-ending 429).
    const mins = Math.ceil(cdMin)
    const span = mins >= 60 ? `${Math.floor(mins / 60)}h ${mins % 60}m` : `${mins}m`
    const res = errorResponse(429, `Amazon Quick usage limit reached; this account resets in ${span} (session window). Failing fast so another account can serve the request.`, 'rate_limit_error')
    // belt and suspenders: magpie also honors Retry-After in both its quota
    // and rate rest paths, and a header survives error translation better
    // than body wording
    res.headers.set('retry-after', String(mins * 60))
    return res
  }
  let socket
  try {
    socket = await socketFor(outpost, { ...options, effortHeader, log: options?.log })
  } catch (e) {
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }
  if (review?.prime) {
    // The policy exceeded one frame: prime it into a dedicated thread's
    // history (a few plain turns), then review on that thread.
    try {
      review = await materializeReview(socket, review, base, signal)
    } catch (e) {
      return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
    }
  }

  const noteQuota = (e) => {
    if (e?.kind === 'quota' || e?.status === 429) quotaMarkExhausted(identity, () => outpost.session(), options?.log)
    return e
  }
  const it = runTurn({ outpost, socket, body, systemText, base, review, sessions, options, signal })

  // Pull up to the first real output before answering so auth/session failures
  // surface as real HTTP statuses (magpie marks the account on 401, fails over
  // on 429/5xx) instead of a 200 stream that dies immediately. For streaming,
  // headerWaitMs caps that wait: agentic turns can sit silent while Quick runs
  // its own tools, and the gateway only waits so long for response headers.
  // pull() keeps at most one it.next() in flight — after the deadline the same
  // pending promise is what items awaits, never a second next() call.
  const headerWaitMs = msOpt(options?.headerWaitMs, 10_000)
  const giveUpAt = body.stream && headerWaitMs > 0 ? Date.now() + headerWaitMs : 0
  const pre = []
  let done = false
  let meta
  let pendingNext = null
  const pull = () => {
    if (!pendingNext) {
      const p = it.next()
      pendingNext = p
      const clear = () => { if (pendingNext === p) pendingNext = null }
      p.then(clear, clear)
    }
    return pendingNext
  }
  try {
    for (;;) {
      let r
      if (giveUpAt) {
        const remaining = giveUpAt - Date.now()
        if (remaining <= 0) break
        let timer
        const timedOut = new Promise((res) => { timer = setTimeout(() => res(HDR_WAIT), remaining) })
        r = await Promise.race([pull(), timedOut])
        clearTimeout(timer)
        if (r === HDR_WAIT) break
      } else {
        r = await pull()
      }
      if (r.done) { done = true; break }
      if (r.value.type === 'meta') { meta = r.value; continue }
      if (r.value.type === 'note') { pre.push(r.value); continue } // notes ride inside the stream, don't commit headers alone
      pre.push(r.value)
      break
    }
  } catch (e) {
    noteQuota(e)
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }
  const items = (async function* () {
    yield* pre
    if (done) return
    for (;;) {
      const r = await pull()
      if (r.done) return
      yield r.value
    }
  })()

  const created = Math.floor(Date.now() / 1000)
  const id = makeId(meta?.threadId)

  if (!body.stream) {
    let text = ''
    let reasoning = ''
    const calls = []
    let usage
    try {
      for await (const o of items) {
        if (o.type === 'text') text += o.text
        else if (o.type === 'reasoning') reasoning += o.text
        else if (o.type === 'tool_call') calls.push({ id: o.id, type: 'function', function: { name: o.name, arguments: o.arguments } })
        else if (o.type === 'usage') usage = addUsage(usage, o.usage)
      }
    } catch (e) {
    noteQuota(e)
      return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
    }
    const message = { role: 'assistant', content: text }
    if (reasoning) message.reasoning_content = reasoning
    if (calls.length) message.tool_calls = calls
    return new Response(
      JSON.stringify({
        id, object: 'chat.completion', created, model,
        choices: [{ index: 0, message, finish_reason: calls.length ? 'tool_calls' : 'stop' }],
        usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )
  }

  const enc = new TextEncoder()
  const includeUsage = !!body.stream_options?.include_usage
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`))
      const chunk = (delta, finish = null) => send({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta, finish_reason: finish }] })
      let sentRole = false
      const role = () => {
        if (!sentRole) {
          sentRole = true
          chunk({ role: 'assistant', content: '' })
        }
      }
      let toolIndex = 0
      let usage
      try {
        for await (const o of items) {
          if (o.type === 'text') {
            role()
            chunk({ content: o.text })
          } else if (o.type === 'reasoning') {
            role()
            chunk({ reasoning_content: o.text })
          } else if (o.type === 'tool_call') {
            role()
            chunk({ tool_calls: [{ index: toolIndex++, id: o.id, type: 'function', function: { name: o.name, arguments: o.arguments } }] })
          } else if (o.type === 'usage') usage = addUsage(usage, o.usage)
          else if (o.type === 'note') {
            role()
            chunk({ content: `\n\n> ${o.text}` })
          }
        }
        role()
        chunk({}, toolIndex > 0 ? 'tool_calls' : 'stop')
        if (includeUsage) send({ id, object: 'chat.completion.chunk', created, model, choices: [], usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } })
        controller.enqueue(enc.encode('data: [DONE]\n\n'))
      } catch (e) {
        // a quota failure mid-stream must mark the account just like the
        // pre-header and non-stream paths do — otherwise the next request
        // burns a whole upstream turn before failing the same way
        noteQuota(e)
        send({ error: { message: e?.message ?? String(e), type: errorType(e), code: httpStatus(e) } })
        controller.enqueue(enc.encode('data: [DONE]\n\n'))
      } finally {
        controller.close()
      }
    },
    cancel() {
      items.return?.()
      it.return?.()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' },
  })
}

/** /v1/models in OpenAI shape. */
export function handleModels() {
  return new Response(JSON.stringify({ object: 'list', data: MODELS }), {
    headers: { 'content-type': 'application/json' },
  })
}

export { AGENT_MODES }

// --- advanced plan-gating fallback (#23) ----------------------------------
// advanced sits behind the advancedModeEnabled entitlement; plans without
// it (Lite free trial) get a fast generic stream error instead of a turn.
// Detect that exact signature, retry once as smart, and cache the downgrade
// per account when smart succeeds — magpie's all-model probe then passes
// instead of cooling the whole account down over one unsupported tier.
const advancedDowngrades = new Set()

export async function handleChatCompletions(args) {
  const identity = args?.options?.identity || 'local'
  const first = await runChatCompletions(args)
  const model = typeof args?.body?.model === 'string' ? args.body.model.trim().toLowerCase() : ''
  const bare = model.includes('/') ? model.split('/').pop() : model
  if (bare !== 'advanced' || first.status !== 502 || advancedDowngrades.has(identity)) return first
  const text = await first.text().catch(() => '')
  const again = () => new Response(text, { status: first.status, headers: first.headers })
  if (!text.includes('Something went wrong')) return again()
  const retry = await runChatCompletions({ ...args, body: { ...args.body, model: 'smart' } })
  if (retry.status !== 200) return again()
  advancedDowngrades.add(identity)
  args?.options?.log?.('[quick] advanced unsupported on this plan — downgraded to smart (cached for this account)')
  return retry
}
