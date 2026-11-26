// OpenAI ⇄ Quick message rendering + model/mode mapping.
// Conversation history travels in text. The application instructions travel
// separately in Quick's per-turn custom_prompt, which the desktop backend
// actually passes to the model.

/** Text content of one content-part object (OpenAI/Anthropic/Responses shapes). */
function partText(p) {
  if (typeof p === 'string') return p
  const t = p?.type
  if (t === 'text' || t === 'input_text' || t === 'output_text') return p.text ?? ''
  if (t === 'image_url' || t === 'input_image' || t === 'image') return '[image]'
  if (t === 'file' || t === 'input_file') return '[file omitted]'
  if (t === 'tool_result') return textOf(p.content) // string or nested parts
  if (t === 'tool_use') {
    const args = typeof p.input === 'string' ? p.input : JSON.stringify(p.input ?? {})
    return `[called tool ${p.name ?? 'tool'}(${args})]`
  }
  return '' // thinking/redacted_thinking/audio/etc carry no user-visible text
}

/** Text content of an OpenAI message (string or parts array). */
export function textOf(content) {
  if (content == null) return ''
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map(partText).filter(Boolean).join('\n')
  }
  return String(content)
}

const DATA_URL = /^data:(image\/[\w.+-]+);base64,(.+)$/s

/** One image part as {mimeType, data(base64)} or {url} when it isn't inline data. */
function partImage(p) {
  const t = p?.type
  if (t !== 'image_url' && t !== 'input_image' && t !== 'image') return null
  const src = p.source
  if (src?.type === 'base64' && typeof src.data === 'string') return { mimeType: src.media_type || 'image/png', data: src.data }
  const url = typeof p.image_url === 'string' ? p.image_url : p.image_url?.url ?? p.url ?? (typeof p.data === 'string' && p.mimeType ? `data:${p.mimeType};base64,${p.data}` : '')
  const m = DATA_URL.exec(String(url ?? ''))
  return m ? { mimeType: m[1], data: m[2].replace(/\s+/g, '') } : { url: String(url ?? '') }
}

/** Images in a message's content, including those nested in tool_result parts. */
export function imagesOf(content) {
  if (!Array.isArray(content)) return []
  return content.flatMap((p) => (p?.type === 'tool_result' ? imagesOf(p.content) : [partImage(p)].filter(Boolean)))
}

const IMAGE_ENTRY_OVERHEAD = 256

// Cloud-upload caps for images too big for the inline frame (per turn).
export const MAX_UPLOAD_IMAGE_BYTES = 3 * 1024 * 1024
export const MAX_UPLOADS = 4
export const MAX_UPLOAD_TOTAL_BYTES = 5 * 1024 * 1024

const imageExt = (mimeType) => mimeType?.split('/')[1]?.replace('jpeg', 'jpg').replace(/[^a-z0-9]/gi, '') || 'img'
const decodedSize = (data) => {
  const pad = data.endsWith('==') ? 2 : data.endsWith('=') ? 1 : 0
  return Math.floor((data.length * 3) / 4) - pad
}

/**
 * Pick images from `messages` for user_message.attachments. Quick counts
 * inline attachment data against the same ~64 KB frame as the text, so only
 * images whose base64 fits `maxBytes` go inline; the rest (and non-inline
 * URLs) are counted in `skipped`. With `{upload:true}` (direct/cloud
 * sessions), inline-data images that don't fit instead queue in `toUpload`
 * for lib/attach.mjs, subject to the upload caps — anything beyond them
 * still lands in `skipped`. Names are numbered across inline + upload.
 */
export function pickImages(messages, maxBytes, { upload = false } = {}) {
  const attachments = []
  const toUpload = []
  let bytes = 0
  let skipped = 0
  let seq = 0
  let uploadCount = 0
  let uploadBytes = 0
  for (const m of messages) {
    for (const img of imagesOf(m.content)) {
      const cost = (img.data?.length ?? Infinity) + IMAGE_ENTRY_OVERHEAD
      if (img.data && bytes + cost <= maxBytes) {
        attachments.push({ type: 'image', name: `image-${++seq}.${imageExt(img.mimeType)}`, size: decodedSize(img.data), mimeType: img.mimeType, data: img.data })
        bytes += cost
        continue
      }
      const decoded = img.data ? decodedSize(img.data) : Infinity
      if (upload && img.data && decoded <= MAX_UPLOAD_IMAGE_BYTES && uploadCount < MAX_UPLOADS && uploadBytes + decoded <= MAX_UPLOAD_TOTAL_BYTES) {
        toUpload.push({ mimeType: img.mimeType, data: img.data, name: `image-${++seq}.${imageExt(img.mimeType)}` })
        uploadCount++
        uploadBytes += decoded
        continue
      }
      skipped++
    }
  }
  return { attachments, bytes, skipped, toUpload }
}

/**
 * Canonical role across client dialects: some agents/gateways emit 'ai',
 * 'human', 'observation' etc. instead of the OpenAI names. Thread-reuse
 * prefix matching and transcript rendering both key off this.
 */
export function canonRole(role) {
  const r = String(role ?? '').toLowerCase()
  if (r === 'assistant' || r === 'ai' || r === 'bot' || r === 'model' || r === 'assistant_message') return 'assistant'
  if (r === 'user' || r === 'human' || r === 'input') return 'user'
  if (r === 'tool' || r === 'tool_result' || r === 'observation' || r === 'tool_output') return 'tool'
  return r
}

/** Agent modes the desktop client can send as `agent_mode`. */
export const AGENT_MODES = Object.freeze(['fast', 'balanced', 'smart', 'advanced'])

export const MODELS = Object.freeze(
  AGENT_MODES.map((mode) => ({
    id: mode,
    object: 'model',
    created: 0,
    owned_by: 'aws-quick',
    name: { fast: 'Quick Fast', balanced: 'Quick Balanced', smart: 'Quick Smart', advanced: 'Quick Advanced' }[mode],
  })),
)

/** Map a requested model id to an agent_mode; unknown → 'smart'. */
export function modeFor(model) {
  const m = String(model ?? '').trim().toLowerCase()
  const bare = m.includes('/') ? m.split('/').pop() : m
  return AGENT_MODES.includes(bare) ? bare : 'smart'
}

/** thinking_effort levels the client understands (off/low/medium/high/max). */
export const EFFORT_LEVELS = Object.freeze(['off', 'low', 'medium', 'high', 'max'])

/**
 * thinking_effort from the X-Magpie-Effort header or body.reasoning_effort.
 * Returns undefined when unset/invalid (server then uses its default).
 */
export function effortFor(effortHeader, body) {
  const raw = effortHeader ?? body?.reasoning_effort ?? body?.thinking_effort
  const v = typeof raw === 'string' ? raw.trim().toLowerCase() : ''
  if (EFFORT_LEVELS.includes(v)) return v
  // magpie/OpenAI spellings → quick levels
  const map = { none: 'off', minimal: 'low', low: 'low', medium: 'medium', med: 'medium', high: 'high', xhigh: 'max', max: 'max' }
  return map[v] ?? (v === 'enabled' || v === 'true' ? 'high' : v === 'disabled' || v === 'false' ? 'off' : undefined)
}

export function renderToolCalls(calls) {
  if (!Array.isArray(calls) || !calls.length) return ''
  return calls
    .map((c) => {
      const name = c?.function?.name ?? c?.name ?? 'tool'
      const args = c?.function?.arguments ?? c?.arguments ?? ''
      return `[called tool ${name}(${typeof args === 'string' ? args : JSON.stringify(args)})]`
    })
    .join('\n')
}

const te = new TextEncoder()
export const byteLen = (s) => te.encode(String(s ?? '')).length

/** user_message.text is hard-capped at 64 KB server-side; stay under it. */
export const UPSTREAM_TEXT_LIMIT = 60 * 1024
const SYSTEM_CAP = 24 * 1024
const USER_TURN_CAP = 20 * 1024
const TOOL_RESULT_CAP = 16 * 1024
const TRUNC_MARK = '\n…[truncated]'

function fitBytes(s, max, measure = byteLen) {
  if (measure(s) <= max) return s
  let lo = 0, hi = s.length
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (measure(s.slice(0, mid)) <= max) lo = mid
    else hi = mid - 1
  }
  if (lo && /[\uD800-\uDBFF]/.test(s[lo - 1])) lo--
  return s.slice(0, lo)
}

/** Truncate to a byte budget, appending a marker. */
export function capText(s, maxBytes, measure = byteLen) {
  const str = String(s ?? '')
  if (measure(str) <= maxBytes) return str
  const room = maxBytes - measure(TRUNC_MARK)
  if (room <= 0) return fitBytes(str, maxBytes, measure)
  return fitBytes(str, room, measure) + TRUNC_MARK
}

const MIDDLE_MARK = '\n…[middle truncated]…\n'

/**
 * Cap a user message keeping head AND tail. The end of a message is often
 * what matters — the question after a pasted log, the last items of a task
 * brief — so the current turn and a pinned opening message are cut in the
 * middle. (System instructions keep capText's placement.)
 */
export function capMiddle(s, maxBytes) {
  const str = String(s ?? '')
  if (byteLen(str) <= maxBytes) return str
  const room = maxBytes - byteLen(MIDDLE_MARK)
  if (room < 512) return capText(str, maxBytes)
  const head = fitBytes(str, Math.floor(room * 0.6))
  return head + MIDDLE_MARK + tailBytes(str, room - byteLen(head))
}

export function frameBytes(threadId, payload) {
  return byteLen(JSON.stringify({ type: 'user_message', payload: { thread_id: threadId, ...payload } }))
}

const escapedBytes = text => byteLen(JSON.stringify(text)) - 2

// On local outpost, Smart retrieved labels from custom_prompt at both 124 B
// and 40 KiB, and missed both labels in system_prompt at either size. Keep
// the persona override small; the caller's instructions appear exactly once.
export function setInstructions(payload, system, custom, threadId) {
  if (!system) {
    if (custom) payload.custom_prompt = custom
    return false
  }
  payload.system_prompt = 'Follow the application instructions supplied for this turn.'
  const prefix = custom ? custom + '\n\n' : ''
  payload.custom_prompt = prefix
  // An omitted thread ID requests lossless placement. The permission reviewer
  // then checks the whole frame and rejects an oversize input before I/O.
  const instruction = threadId === undefined ? system
    : capText(system, Math.max(0, UPSTREAM_TEXT_LIMIT - frameBytes(threadId, payload)), escapedBytes)
  payload.custom_prompt += instruction
  return instruction !== system
}

const STALE_TOOL_CAP = 2 * 1024

/** Longest suffix of `s` within `max` bytes. */
function tailBytes(s, max) {
  if (max <= 0) return ''
  let lo = 0, hi = s.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (byteLen(s.slice(mid)) <= max) hi = mid
    else lo = mid + 1
  }
  return s.slice(lo)
}

/**
 * Shrink a tool result to `max` bytes keeping head + tail (file headers and
 * trailing errors both matter) and telling the model how to get the rest —
 * a silent cut makes it guess, a vanished result makes it re-run blindly.
 */
function shrinkTool(body, max) {
  const total = byteLen(body)
  if (total <= max) return body
  const mark = `\n…[${total}-byte result, middle omitted to fit Quick's frame — re-run the tool with a narrower range if you need it]…\n`
  const room = max - byteLen(mark)
  if (room < 256) return `[${total}-byte result omitted to fit Quick's frame — re-run the tool if you need it]`
  const head = fitBytes(body, Math.floor((room * 2) / 3))
  return head + mark + tailBytes(body, room - byteLen(head))
}

const toolHead = (m) => `[tool_result ${[m.name, m.tool_call_id].filter(Boolean).join(' ')}]`
const isTool = (m) => canonRole(m.role) === 'tool' || m.role === 'function'
const renderBlock = (b) => (b.tool ? `${b.head}\n${b.body}` : b.body)

/**
 * Fit blocks into `maxBytes`. Conversation text is fixed; tool-result bodies
 * share what's left by water-filling (small results stay whole, large ones
 * split the remainder evenly), so parallel results all survive partially
 * instead of the oldest vanishing. Only if text alone overflows are the
 * oldest blocks dropped; the last block is never dropped.
 */
function fitBlocks(blocks, maxBytes, cut, { pinFirst = false } = {}) {
  let avail = maxBytes - blocks.reduce((n, b) => n + 2 + (b.tool ? byteLen(b.head) + 1 : byteLen(b.body)), 0)
  const tools = blocks.filter((b) => b.tool).sort((a, b) => byteLen(a.body) - byteLen(b.body))
  let left = tools.length
  for (const b of tools) {
    const share = Math.max(0, Math.floor(avail / left--))
    if (byteLen(b.body) > share) { b.body = shrinkTool(b.body, share); cut() }
    avail -= byteLen(b.body)
  }
  const out = blocks.map(renderBlock)
  let used = out.reduce((n, s) => n + byteLen(s) + 2, 0)
  // pinFirst: the opening message (a compaction summary or the original task)
  // is what the model can least afford to lose — drop from the second block
  // on, and only shrink the first once everything between is gone.
  const pin = pinFirst && out.length > 2 ? 1 : 0
  let dropped = 0
  while (used > maxBytes && pin + dropped < out.length - 1) { used -= byteLen(out[pin + dropped++]) + 2; cut() }
  const marker = `[${dropped} earlier message(s) omitted]`
  if (pin && dropped) used += byteLen(marker) + 2
  const kept = pin
    ? [out[0], ...(dropped ? [marker] : []), ...out.slice(1 + dropped)]
    : out.slice(dropped)
  if (pin && used > maxBytes) {
    // squeeze the pinned head to what's left beside the rest
    const room = Math.max(512, maxBytes - (used - byteLen(kept[0])))
    if (room < byteLen(kept[0])) { used -= byteLen(kept[0]); kept[0] = capMiddle(kept[0], room); used += byteLen(kept[0]); cut() }
  }
  if (used > maxBytes && kept.length) {
    const others = used - byteLen(kept[kept.length - 1]) - 2
    kept[kept.length - 1] = capMiddle(kept[kept.length - 1], Math.max(0, maxBytes - others - 2))
    cut()
  }
  // a pinned drop is marked inline; callers only label head-of-list drops
  return { kept, dropped: pin ? 0 : dropped }
}

/**
 * Claude Code re-injects whole files as <system-reminder> blocks inside user
 * messages. In history they are stale context, not conversation: squeeze
 * each oversized one like a tool result so it can't crowd out real turns.
 */
const REMINDER_RE = /<system-reminder>([\s\S]*?)<\/system-reminder>/g
const REMINDER_CAP = 2 * 1024
function squeezeReminders(text, cut) {
  if (!text.includes('<system-reminder>')) return text
  return text.replace(REMINDER_RE, (all, inner) => {
    if (byteLen(inner) <= REMINDER_CAP) return all
    cut()
    return `<system-reminder>${shrinkTool(inner, REMINDER_CAP)}</system-reminder>`
  })
}

function toolBlock(m, cap, cut) {
  let body = textOf(m.content)
  if (byteLen(body) > cap) { body = shrinkTool(body, cap); cut() }
  return { tool: true, head: toolHead(m), body }
}

/** One transcript line for a non-tool message in the rendered history. */
function transcriptLine(m, cut = () => {}) {
  const text = squeezeReminders(textOf(m.content), cut)
  const role = canonRole(m.role)
  if (role === 'assistant') {
    const calls = renderToolCalls(m.tool_calls)
    return `assistant: ${[text, calls].filter(Boolean).join('\n')}`
  }
  return `${role}: ${text}`
}

const HISTORY_HEAD_RESERVE = 96

/**
 * Byte cap for the current user turn: at least USER_TURN_CAP, up to 60% of
 * the budget — a long final message (a task brief, a pasted log) is usually
 * the thing to answer; older history yields first.
 */
function lastTurnCap(budget) {
  return Number.isFinite(budget) ? Math.max(USER_TURN_CAP, Math.floor(budget * 0.6)) : Infinity
}

/**
 * Full-transcript text for a NEW thread: system prompt + rendered history +
 * the final turn. Everything the model needs in one user_message.
 * Tool results answered before the latest assistant turn are stale and
 * squeezed to STALE_TOOL_CAP; the rest share the budget via fitBlocks.
 * `stats` is mutated with {truncated:true} when anything was cut.
 */
export function renderTranscript(messages, systemText, { maxBytes = Infinity, stats } = {}) {
  const cut = () => { if (stats) stats.truncated = true }
  const parts = []
  if (systemText) {
    if (byteLen(systemText) > SYSTEM_CAP) { systemText = capText(systemText, SYSTEM_CAP); cut() }
    parts.push(`[system instructions]\n${systemText}`)
  }
  let lastAsst = -1
  messages.forEach((m, i) => { if (canonRole(m.role) === 'assistant') lastAsst = i })
  const sysBytes = parts.reduce((n, s) => n + byteLen(s) + 2, 0)
  const budget = maxBytes - sysBytes - (messages.length > 1 ? HISTORY_HEAD_RESERVE : 0)
  const turnCap = lastTurnCap(budget)
  const freshToolCap = Number.isFinite(maxBytes) ? maxBytes : TOOL_RESULT_CAP
  const blocks = messages.map((m, i) => {
    if (isTool(m)) return toolBlock(m, i < lastAsst ? STALE_TOOL_CAP : freshToolCap, cut)
    if (i === messages.length - 1 && canonRole(m.role) === 'user') {
      let t = textOf(m.content)
      if (byteLen(t) > turnCap) { t = capMiddle(t, turnCap); cut() }
      return { tool: false, body: t || '(empty)' }
    }
    return { tool: false, body: transcriptLine(m, cut) }
  })
  if (!blocks.length) blocks.push({ tool: false, body: '(empty)' })
  const pinFirst = canonRole(messages[0]?.role) === 'user'
  const { kept, dropped } = fitBlocks(blocks, budget, cut, { pinFirst })
  const last = kept.pop()
  if (kept.length || dropped) {
    const head = dropped
      ? `[conversation so far — ${dropped} earlier message(s) omitted]\n`
      : '[conversation so far]\n'
    parts.push(head + kept.join('\n\n'))
  }
  parts.push(last)
  return parts.join('\n\n')
}

/**
 * Delta text for a REUSED thread: only the new user/tool content.
 * With a finite `maxBytes`, fresh tool results skip the 16 KB pre-cap so
 * `fitBlocks` can water-fill the full remaining frame budget across single or
 * parallel results — the current turn's question is never sacrificed for tool
 * output.
 */
export function renderDelta(delta, { maxBytes = Infinity, stats } = {}) {
  const cut = () => { if (stats) stats.truncated = true }
  // Foreign turns (#16): a gateway failover can route mid-conversation to
  // another account and back, so the delta may contain assistant replies (and
  // their tool results) this thread never saw. Render the delta transcript-
  // style — user:/assistant: prefixes plus a header note — so the model
  // absorbs it as its own conversation history. Without foreign content the
  // rendering is byte-identical to before: bare user text, tool blocks.
  const foreign = delta.some((m) => canonRole(m.role) === 'assistant')
  let lastAsst = -1
  if (foreign) delta.forEach((m, i) => { if (canonRole(m.role) === 'assistant') lastAsst = i })
  const freshToolCap = Number.isFinite(maxBytes) ? maxBytes : TOOL_RESULT_CAP
  const blocks = []
  for (let i = 0; i < delta.length; i++) {
    const m = delta[i]
    if (isTool(m)) {
      blocks.push(toolBlock(m, i < lastAsst ? STALE_TOOL_CAP : freshToolCap, cut))
      continue
    }
    const role = canonRole(m.role)
    if (role === 'assistant') {
      blocks.push({ tool: false, body: transcriptLine(m, cut) })
      continue
    }
    if (role !== 'user') continue
    const t = textOf(m.content)
    if (t) blocks.push({ tool: false, body: t, user: true })
  }
  // the newest user message is the current turn: it gets the larger cap
  const lastUser = blocks.findLastIndex((b) => !b.tool)
  blocks.forEach((b, i) => {
    if (b.tool) return
    const cap = i === lastUser ? lastTurnCap(maxBytes - 48) : USER_TURN_CAP
    if (byteLen(b.body) > cap) { b.body = capMiddle(b.body, cap); cut() }
    // in a foreign delta every earlier user message is history, not the
    // current turn — transcript-prefix it so it can't read as a new instruction
    if (foreign && b.user && i !== lastUser) b.body = `user: ${b.body}`
  })
  if (!blocks.length) return ''
  const { kept, dropped } = fitBlocks(blocks, maxBytes - 48, cut)
  const joined = dropped ? `[${dropped} earlier message(s) omitted]\n` + kept.join('\n\n') : kept.join('\n\n')
  if (!foreign) return joined
  return `[since your last reply the conversation continued elsewhere — everything down to the final message already happened in this conversation, treat it as if it happened here]\n${joined}`
}
