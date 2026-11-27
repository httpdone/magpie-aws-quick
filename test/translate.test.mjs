import { test } from 'node:test'
import assert from 'node:assert/strict'
import { textOf, modeFor, effortFor, renderTranscript, renderDelta, byteLen, capText, UPSTREAM_TEXT_LIMIT, imagesOf, pickImages, MAX_UPLOAD_IMAGE_BYTES, MAX_UPLOADS, MAX_UPLOAD_TOTAL_BYTES } from '../lib/translate.mjs'

test('textOf handles strings, parts arrays, and junk', () => {
  assert.equal(textOf('hi'), 'hi')
  assert.equal(textOf([{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }]), 'a\nb')
  assert.equal(textOf(null), '')
  assert.equal(textOf([{ type: 'image_url', image_url: { url: 'x' } }]), '[image]')
  assert.equal(textOf(42), '42')
})

test('textOf understands Anthropic/Responses-style content parts', () => {
  // Responses API spellings
  assert.equal(textOf([{ type: 'input_text', text: 'hello' }]), 'hello')
  assert.equal(textOf([{ type: 'output_text', text: 'world' }]), 'world')
  // Anthropic tool blocks inside content arrays
  assert.equal(textOf([{ type: 'tool_result', content: 'tool output' }]), 'tool output')
  assert.equal(
    textOf([{ type: 'tool_result', content: [{ type: 'text', text: 'nested' }] }]),
    'nested',
  )
  assert.equal(
    textOf([{ type: 'tool_use', name: 'Read', input: { path: '/a' } }]),
    '[called tool Read({"path":"/a"})]',
  )
  // reasoning blocks contribute nothing
  assert.equal(
    textOf([{ type: 'thinking', thinking: 'secret' }, { type: 'text', text: 'answer' }]),
    'answer',
  )
})

test('modeFor maps model ids to agent modes', () => {
  assert.equal(modeFor('fast'), 'fast')
  assert.equal(modeFor('balanced'), 'balanced')
  assert.equal(modeFor('smart'), 'smart')
  assert.equal(modeFor('advanced'), 'advanced')
  assert.equal(modeFor('provider/fast'), 'fast')
  assert.equal(modeFor('AWS Quick Smart'), 'smart')
  assert.equal(modeFor('gpt-5'), 'smart') // unknown → default
  assert.equal(modeFor(undefined), 'smart')
  assert.equal(modeFor(''), 'smart')
})

test('effortFor maps header/body spellings to quick levels', () => {
  assert.equal(effortFor('high', {}), 'high')
  assert.equal(effortFor(undefined, { reasoning_effort: 'low' }), 'low')
  assert.equal(effortFor(undefined, { thinking_effort: 'max' }), 'max')
  assert.equal(effortFor('xhigh', {}), 'max')
  assert.equal(effortFor('none', {}), 'off')
  assert.equal(effortFor('bogus', {}), undefined)
  assert.equal(effortFor(undefined, {}), undefined)
})

test('renderTranscript: system + history + raw final user turn', () => {
  const messages = [
    { role: 'user', content: 'first' },
    { role: 'assistant', content: 'reply one' },
    { role: 'user', content: 'second' },
  ]
  const out = renderTranscript(messages, 'be terse')
  assert.ok(out.includes('[system instructions]\nbe terse'))
  assert.ok(out.includes('[conversation so far]'))
  assert.ok(out.includes('user: first'))
  assert.ok(out.includes('assistant: reply one'))
  // last user message goes in raw, no role prefix
  assert.ok(out.endsWith('\n\nsecond'))
  assert.ok(!out.includes('user: second'))
})

test('renderTranscript without system/history is just the user text', () => {
  assert.equal(renderTranscript([{ role: 'user', content: 'hello' }], ''), 'hello')
})

test('renderDelta renders only new user/tool content', () => {
  const delta = [
    { role: 'user', content: 'next question' },
    { role: 'tool', tool_call_id: 'c1', name: 'read', content: 'file body' },
  ]
  const out = renderDelta(delta)
  assert.ok(out.includes('next question'))
  assert.ok(out.includes('[tool_result read c1]'))
  assert.ok(out.includes('file body'))
})

test('capText truncates by bytes with marker and respects budget', () => {
  const s = 'x'.repeat(10 * 1024)
  const out = capText(s, 4096)
  assert.ok(byteLen(out) <= 4096)
  assert.ok(out.endsWith('[truncated]'))
  assert.equal(capText('short', 4096), 'short')
  // multi-byte chars: byte budget must hold even for CJK
  const cjk = capText('汉'.repeat(5000), 4096)
  assert.ok(byteLen(cjk) <= 4096)
  assert.ok(cjk.includes('truncated'))
})

test('renderTranscript keeps the opening user message, drops the middle, keeps final user turn', () => {
  const messages = [
    { role: 'user', content: 'OLD-QUESTION-' + 'a'.repeat(3000) },
    { role: 'assistant', content: 'OLD-ANSWER-' + 'b'.repeat(3000) },
    { role: 'user', content: 'RECENT-' + 'c'.repeat(2000) },
    { role: 'assistant', content: 'RECENT-ANSWER-' + 'd'.repeat(2000) },
    { role: 'user', content: 'FINAL-QUESTION' },
  ]
  const stats = {}
  const out = renderTranscript(messages, 'sys', { maxBytes: 8000, stats })
  assert.ok(stats.truncated)
  assert.ok(out.endsWith('FINAL-QUESTION')) // current turn never dropped
  assert.ok(out.includes('OLD-QUESTION')) // opening message (task / compaction summary) is pinned
  assert.ok(!out.includes('OLD-ANSWER')) // the middle goes first
  assert.ok(out.indexOf('OLD-QUESTION') < out.indexOf('earlier message(s) omitted'))
  assert.ok(byteLen(out) <= 8000)
})

test('renderTranscript caps oversized system prompt and user turn', () => {
  const messages = [{ role: 'user', content: 'u'.repeat(30 * 1024) }]
  const stats = {}
  const out = renderTranscript(messages, 's'.repeat(40 * 1024), { maxBytes: UPSTREAM_TEXT_LIMIT, stats })
  assert.ok(stats.truncated)
  assert.ok(out.includes('[truncated]'))
  assert.ok(byteLen(out) <= UPSTREAM_TEXT_LIMIT)
  assert.ok(out.includes('[middle truncated]')) // user turn cut in the middle…
  assert.ok(out.endsWith('u')) // …so its end survives
})

test('renderDelta caps oversized tool results per block', () => {
  const delta = [
    { role: 'tool', tool_call_id: 'c1', name: 'read', content: 'F'.repeat(30 * 1024) },
    { role: 'user', content: 'next' },
  ]
  const stats = {}
  const out = renderDelta(delta, { stats })
  assert.ok(stats.truncated)
  assert.ok(out.includes('[tool_result read c1]'))
  assert.ok(out.includes('next'))
  assert.ok(byteLen(out) < 20 * 1024) // 16KB cap + small user msg
})

test('renderDelta shares the budget across parallel tool results instead of dropping the oldest', () => {
  const delta = [
    { role: 'tool', tool_call_id: 'c1', name: 'a', content: 'A'.repeat(10 * 1024) },
    { role: 'tool', tool_call_id: 'c2', name: 'b', content: 'B'.repeat(10 * 1024) },
    { role: 'tool', tool_call_id: 'c3', name: 'c', content: 'C'.repeat(10 * 1024) },
    { role: 'user', content: 'final question' },
  ]
  const stats = {}
  const out = renderDelta(delta, { maxBytes: 20 * 1024, stats })
  assert.ok(stats.truncated)
  assert.ok(byteLen(out) <= 20 * 1024)
  assert.ok(out.endsWith('final question'))
  for (const h of ['[tool_result a c1]', '[tool_result b c2]', '[tool_result c c3]']) assert.ok(out.includes(h), h)
  assert.ok(!out.includes('earlier message(s) omitted'))
  assert.equal((out.match(/middle omitted/g) || []).length, 3)
})

test('renderDelta keeps small results whole and squeezes only the large one', () => {
  const delta = [
    { role: 'tool', tool_call_id: 'c1', name: 'small', content: 'tiny output' },
    { role: 'tool', tool_call_id: 'c2', name: 'big', content: 'HEAD' + 'x'.repeat(40 * 1024) + 'TAIL-ERROR' },
  ]
  const out = renderDelta(delta, { maxBytes: 8 * 1024 })
  assert.ok(byteLen(out) <= 8 * 1024)
  assert.ok(out.includes('tiny output'))
  assert.ok(out.includes('HEAD') && out.includes('TAIL-ERROR')) // head + tail both survive
  assert.ok(out.includes('re-run the tool'))
})

test('renderDelta still drops oldest text when conversation text alone overflows', () => {
  const delta = [
    { role: 'user', content: 'OLD-' + 'a'.repeat(6000) },
    { role: 'user', content: 'NEW-QUESTION' },
  ]
  const out = renderDelta(delta, { maxBytes: 4096 })
  assert.ok(byteLen(out) <= 4096)
  assert.ok(!out.includes('OLD-'))
  assert.ok(out.includes('earlier message(s) omitted'))
  assert.ok(out.endsWith('NEW-QUESTION'))
})

test('renderTranscript squeezes stale tool results, keeps fresh ones and all dialogue', () => {
  const messages = [
    { role: 'user', content: 'FIRST-ASK' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'c1', function: { name: 'read', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c1', name: 'read', content: 'STALE' + 's'.repeat(12 * 1024) },
    { role: 'assistant', content: 'MIDDLE-ANSWER', tool_calls: [{ id: 'c2', function: { name: 'read', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c2', name: 'read', content: 'FRESH' + 'f'.repeat(8 * 1024) },
  ]
  const out = renderTranscript(messages, '', { maxBytes: UPSTREAM_TEXT_LIMIT })
  assert.ok(out.includes('FIRST-ASK') && out.includes('MIDDLE-ANSWER'))
  assert.ok(out.includes('STALE') && out.includes('middle omitted')) // stale result squeezed, not dropped
  assert.ok(out.includes('f'.repeat(8 * 1024))) // fresh result fits whole
  assert.ok(byteLen(out) < 16 * 1024)
})

test('imagesOf reads OpenAI, Responses, Anthropic and nested tool_result image shapes', () => {
  const imgs = imagesOf([
    { type: 'image_url', image_url: { url: 'data:image/png;base64,QUJD' } },
    { type: 'input_image', image_url: 'data:image/jpeg;base64,REVG' },
    { type: 'image', source: { type: 'base64', media_type: 'image/webp', data: 'R0hJ' } },
    { type: 'tool_result', content: [{ type: 'image_url', image_url: { url: 'data:image/gif;base64,SktM' } }] },
    { type: 'image_url', image_url: { url: 'https://example.com/a.png' } },
    { type: 'text', text: 'not an image' },
  ])
  assert.deepEqual(imgs.slice(0, 4).map((i) => i.mimeType), ['image/png', 'image/jpeg', 'image/webp', 'image/gif'])
  assert.equal(imgs[0].data, 'QUJD')
  assert.deepEqual(imgs[4], { url: 'https://example.com/a.png' })
  assert.equal(imgs.length, 5)
})

test('pickImages keeps images that fit the byte budget and counts the rest as skipped', () => {
  const small = 'A'.repeat(1000)
  const big = 'B'.repeat(50 * 1024)
  const msgs = [
    { role: 'user', content: [{ type: 'image_url', image_url: { url: `data:image/png;base64,${big}` } }, { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${small}` } }] },
    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'https://example.com/x.png' } }] },
  ]
  const r = pickImages(msgs, 10 * 1024)
  assert.equal(r.attachments.length, 1)
  assert.deepEqual(r.attachments[0], { type: 'image', name: 'image-1.jpg', size: 750, mimeType: 'image/jpeg', data: small })
  assert.equal(r.skipped, 2)
  assert.ok(r.bytes >= small.length && r.bytes <= 10 * 1024)
  assert.deepEqual(r.toUpload, [])
})

const b64 = (bytes, fill = 7) => Buffer.alloc(bytes, fill).toString('base64')
const imgPart = (data, mime = 'image/png') => ({ type: 'image_url', image_url: { url: `data:${mime};base64,${data}` } })

test('pickImages queues oversized inline images for upload when enabled', () => {
  const big = b64(100 * 1024)   // ~100 KB decoded — far over any inline budget
  const small = b64(400)
  const msgs = [{
    role: 'user',
    content: [imgPart(big), imgPart(small, 'image/jpeg'), { type: 'image_url', image_url: { url: 'https://example.com/x.png' } }],
  }]
  const r = pickImages(msgs, 4 * 1024, { upload: true })
  // the small one still goes inline; numbering continues across both lists
  assert.equal(r.attachments.length, 1)
  assert.equal(r.attachments[0].name, 'image-2.jpg')
  assert.equal(r.attachments[0].data, small)
  assert.equal(r.toUpload.length, 1)
  assert.deepEqual(r.toUpload[0], { mimeType: 'image/png', data: big, name: 'image-1.png' })
  assert.equal(r.skipped, 1) // the remote URL
  assert.equal(r.bytes, small.length + 256)

  // same input without upload → the big image is skipped, not queued
  const r2 = pickImages(msgs, 4 * 1024)
  assert.equal(r2.attachments.length, 1)
  assert.equal(r2.attachments[0].name, 'image-1.jpg')
  assert.deepEqual(r2.toUpload, [])
  assert.equal(r2.skipped, 2)
})

test('pickImages upload caps: count, per-image size and total bytes', () => {
  const oneMiB = b64(1024 * 1024)
  const five = Array.from({ length: MAX_UPLOADS + 1 }, () => imgPart(oneMiB))
  const r = pickImages([{ role: 'user', content: five }], 1024, { upload: true })
  assert.equal(r.toUpload.length, MAX_UPLOADS)
  assert.equal(r.skipped, 1) // the 5th upload is over the per-turn cap

  // a single image over 3 MiB decoded never uploads
  const huge = b64(MAX_UPLOAD_IMAGE_BYTES + 1)
  const r2 = pickImages([{ role: 'user', content: [imgPart(huge)] }], 1024, { upload: true })
  assert.equal(r2.toUpload.length, 0)
  assert.equal(r2.skipped, 1)
  // exactly at the cap is allowed
  const atCap = b64(MAX_UPLOAD_IMAGE_BYTES)
  const r3 = pickImages([{ role: 'user', content: [imgPart(atCap)] }], 1024, { upload: true })
  assert.equal(r3.toUpload.length, 1)

  // total decoded budget: two ~2.6 MiB images exceed 5 MiB together
  const two = b64(Math.floor(MAX_UPLOAD_TOTAL_BYTES / 2) + 1024)
  const r4 = pickImages([{ role: 'user', content: [imgPart(two), imgPart(two)] }], 1024, { upload: true })
  assert.equal(r4.toUpload.length, 1)
  assert.equal(r4.skipped, 1)
})

test('compaction summary survives a tight budget; stale file reminders in history are squeezed', () => {
  const summary = 'SUMMARY-START This session is being continued. Server IP 10.1.2.3. ' + 'x'.repeat(6000) + ' SUMMARY-END'
  const reminder = '<system-reminder>FILE-HEAD ' + 'f'.repeat(30 * 1024) + ' FILE-TAIL</system-reminder>'
  const messages = [
    { role: 'user', content: summary },
    { role: 'assistant', content: 'ok' },
    { role: 'user', content: 'please look ' + reminder },
    { role: 'assistant', content: 'MIDDLE-ANSWER ' + 'm'.repeat(20 * 1024) },
    { role: 'user', content: 'NOW-DO-THIS' },
  ]
  const out = renderTranscript(messages, '', { maxBytes: 20 * 1024 })
  assert.ok(byteLen(out) <= 20 * 1024)
  assert.ok(out.includes('SUMMARY-START') && out.includes('SUMMARY-END'), 'pinned summary kept whole when it fits')
  assert.ok(out.endsWith('NOW-DO-THIS'))
  assert.ok(!out.includes('f'.repeat(4096)), 'file reminder squeezed, not 30 KB')
})

test('renderDelta gives the current turn head and tail, not just the head', () => {
  const big = 'TASK-HEAD ' + 'b'.repeat(40 * 1024) + ' TASK-TAIL'
  const out = renderDelta([{ role: 'user', content: big }], { maxBytes: 60 * 1024 })
  assert.ok(out.includes('TASK-HEAD') && out.includes('TASK-TAIL'))
  assert.ok(byteLen(out) <= 60 * 1024)
})

test('renderDelta: foreign failover turns render as transcript (#16)', () => {
  const delta = [
    { role: 'user', content: 'new decision' },
    { role: 'assistant', content: 'foreign reply', tool_calls: [{ id: 'c1', function: { name: 'Write', arguments: '{}' } }] },
    { role: 'tool', tool_call_id: 'c1', content: 'file written' },
    { role: 'user', content: 'continue' },
  ]
  const out = renderDelta(delta)
  // header note explains the injected history
  assert.ok(out.startsWith('[since your last reply the conversation continued elsewhere'))
  // earlier user messages are transcript-prefixed, assistant lines rendered with calls
  assert.ok(out.includes('user: new decision'))
  assert.ok(out.includes('assistant: foreign reply'))
  assert.ok(out.includes('[called tool Write('))
  assert.ok(out.includes('file written'))
  // the final user message stays bare — it is the current turn, not history
  assert.ok(out.trimEnd().endsWith('continue'))
  assert.ok(!out.includes('user: continue'))
})

test('renderDelta: clean deltas never get the foreign treatment', () => {
  const out = renderDelta([{ role: 'user', content: 'plain question' }])
  assert.equal(out, 'plain question')
})

test('renderDelta lets large single or parallel tool results use the available finite maxBytes budget', () => {
  // Single 40KB file in a 55KB budget stays completely untruncated (no premature 16KB cap).
  const single = [{ role: 'tool', tool_call_id: 'c1', name: 'Read', content: 'A'.repeat(40 * 1024) }]
  const s1 = {}
  const out1 = renderDelta(single, { maxBytes: 55 * 1024, stats: s1 })
  assert.equal(s1.truncated, undefined)
  assert.ok(out1.includes('A'.repeat(40 * 1024)))

  // Two parallel 24KB files in a 55KB budget both survive untruncated.
  const pair = [
    { role: 'tool', tool_call_id: 'c1', name: 'Read', content: 'X'.repeat(24 * 1024) },
    { role: 'tool', tool_call_id: 'c2', name: 'Read', content: 'Y'.repeat(24 * 1024) },
  ]
  const s2 = {}
  const out2 = renderDelta(pair, { maxBytes: 55 * 1024, stats: s2 })
  assert.equal(s2.truncated, undefined)
  assert.ok(out2.includes('X'.repeat(24 * 1024)))
  assert.ok(out2.includes('Y'.repeat(24 * 1024)))

  // 40KB tool result + 20KB user question in a 30KB budget: user stays intact, tool gets ~10KB.
  const q = 'Q'.repeat(20 * 1024)
  const mixed = [
    { role: 'tool', tool_call_id: 'c1', name: 'Read', content: 'HEAD' + 'T'.repeat(40 * 1024) + 'TAIL' },
    { role: 'user', content: q },
  ]
  const s3 = {}
  const out3 = renderDelta(mixed, { maxBytes: 30 * 1024, stats: s3 })
  assert.equal(s3.truncated, true)
  assert.ok(byteLen(out3) <= 30 * 1024)
  assert.ok(out3.endsWith(q), 'user question must stay untruncated')
  assert.ok(out3.includes('HEAD') && out3.includes('TAIL'), 'tool result keeps head + tail in remaining room')
})

test('renderTranscript and renderDelta map tool names without mutating input messages', () => {
  const mapToolName = (n) => (n === 'web_search' ? 'client_web_search' : n)
  const msgs = [
    { role: 'user', content: 'search' },
    { role: 'assistant', content: [{ type: 'tool_use', name: 'web_search', input: { q: 'test' } }], tool_calls: [{ id: 'c1', type: 'function', function: { name: 'web_search', arguments: '{"q":"test"}' } }] },
    { role: 'tool', tool_call_id: 'c1', name: 'web_search', content: 'result body' },
    { role: 'user', content: 'follow up' },
  ]
  const transcript = renderTranscript(msgs, '')
  assert.ok(transcript.includes('[called tool web_search('))
  const mappedTranscript = renderTranscript(msgs, '', { mapToolName })
  assert.ok(mappedTranscript.includes('[called tool client_web_search({"q":"test"})]'))
  assert.ok(mappedTranscript.includes('[tool_result client_web_search c1]'))
  assert.ok(!mappedTranscript.includes('[tool_result web_search c1]'))

  const delta = [
    { role: 'assistant', content: '', tool_calls: [{ id: 'c2', type: 'function', function: { name: 'web_search', arguments: '{"q":"x"}' } }] },
    { role: 'tool', tool_call_id: 'c2', name: 'web_search', content: 'ok' },
    { role: 'user', content: 'next' },
  ]
  const mappedDelta = renderDelta(delta, { mapToolName })
  assert.ok(mappedDelta.includes('[called tool client_web_search({"q":"x"})]'))
  assert.ok(mappedDelta.includes('[tool_result client_web_search c2]'))
  assert.equal(msgs[1].tool_calls[0].function.name, 'web_search')
  assert.equal(msgs[2].name, 'web_search')
  assert.equal(delta[1].name, 'web_search')
})
