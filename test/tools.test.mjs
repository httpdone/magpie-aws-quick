import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeTools, aliasClientTools, aliasToolChoice, buildToolPreamble, createToolCallParser, slimSchema, createSourcesStripper } from '../lib/tools.mjs'
import { byteLen } from '../lib/translate.mjs'

test('normalizeTools keeps only named function tools', () => {
  const tools = normalizeTools([
    { type: 'function', function: { name: 'get_weather', description: 'w', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'no_params' } },
    { type: 'bogus' },
    null,
  ])
  assert.equal(tools.length, 2)
  assert.equal(tools[0].name, 'get_weather')
  assert.equal(tools[1].parameters.type, 'object') // defaulted
})

test('buildToolPreamble embeds specs and forced choice', () => {
  const tools = normalizeTools([{ type: 'function', function: { name: 'f', description: 'd', parameters: { type: 'object', properties: { x: { type: 'string' } } } } }])
  const p = buildToolPreamble(tools, 'required')
  assert.ok(p.includes('<tool_call>{"name":"<tool_name>"'))
  assert.ok(p.includes('- f: d'))
  assert.ok(p.includes('reply with <tool_call> block(s) only'))
  assert.equal(buildToolPreamble([], 'auto'), '')
  const named = buildToolPreamble(tools, { type: 'function', function: { name: 'f' } })
  assert.ok(named.includes('requires the client tool "f"'))
})

test('slimSchema strips annotation keywords, keeps structure', () => {
  const slim = slimSchema({
    type: 'object',
    description: 'verbose docs',
    title: 'T',
    required: ['a'],
    properties: {
      a: { type: 'string', description: 'field doc', enum: ['x', 'y'] },
      b: { type: 'object', properties: { c: { type: 'number', examples: [1] } } },
    },
  })
  assert.equal(slim.description, undefined)
  assert.equal(slim.title, undefined)
  assert.deepEqual(slim.required, ['a'])
  assert.equal(slim.properties.a.description, undefined)
  assert.deepEqual(slim.properties.a.enum, ['x', 'y'])
  assert.equal(slim.properties.b.properties.c.examples, undefined)
})

test('preamble compacts schema annotations and caps giant schemas', () => {
  const bigProps = {}
  for (let i = 0; i < 200; i++) {
    bigProps[`field_${i}`] = { type: 'string', description: 'd'.repeat(80) }
  }
  const tools = normalizeTools([
    { type: 'function', function: { name: 'small', description: 's', parameters: { type: 'object', properties: { x: { type: 'string', description: 'doc' } } } } },
    { type: 'function', function: { name: 'huge', description: 'h', parameters: { type: 'object', properties: bigProps, required: ['field_0'] } } },
  ])
  const p = buildToolPreamble(tools, 'auto')
  // small tool keeps its (slimmed) schema incl. type info
  assert.ok(p.includes('"x":{"type":"string"}'))
  assert.ok(!p.includes('"doc"')) // property description stripped
  // huge tool degrades to bare object + required
  assert.ok(p.includes('"properties":{}'))
  assert.ok(p.includes('"required":["field_0"]'))
  assert.ok(!p.includes('field_1'))
})

test('preamble degrades from full JSON schema to compact typed signatures, then lite', () => {
  // 60 tools with 15 params exceed PREAMBLE_CAP (40KB) in full JSON schema,
  // so buildToolPreamble falls back to compact signatures (preserving param names/types).
  const medium = normalizeTools(Array.from({ length: 60 }, (_, i) => ({
    type: 'function',
    function: {
      name: `tool_${i}`,
      description: `description ${i} `.repeat(30),
      parameters: {
        type: 'object',
        required: ['p0'],
        properties: Object.fromEntries(Array.from({ length: 15 }, (_, j) => [`p${j}`, { type: 'string', description: 'x'.repeat(60) }])),
      },
    },
  })))
  const statsCompact = {}
  const pCompact = buildToolPreamble(medium, 'auto', { stats: statsCompact })
  assert.ok(byteLen(pCompact) < 40 * 1024)
  assert.ok(pCompact.includes('compact signatures'))
  assert.ok(pCompact.includes('- tool_0(p0!: string, p1?: string'))
  assert.ok(!pCompact.includes('"properties"'))
  assert.equal(statsCompact.truncated, true)

  // Truly enormous toolset where even compact signatures exceed PREAMBLE_CAP
  // falls back all the way to lite (name + description).
  const huge = normalizeTools(Array.from({ length: 120 }, (_, i) => ({
    type: 'function',
    function: {
      name: `tool_${i}`,
      description: `description ${i} `.repeat(20),
      parameters: {
        type: 'object',
        properties: Object.fromEntries(Array.from({ length: 20 }, (_, j) => [`param_${j}`, { type: 'string' }])),
      },
    },
  })))
  const statsLite = {}
  const pLite = buildToolPreamble(huge, 'auto', { stats: statsLite })
  assert.ok(byteLen(pLite) < 40 * 1024)
  assert.ok(pLite.includes('parameter schemas omitted'))
  assert.ok(pLite.includes('- tool_0:'))
  assert.equal(statsLite.truncated, true)
})

test('parser extracts blocks split across deltas', () => {
  const p = createToolCallParser()
  let r = p.push('Hello <tool_call>{"name":"get_weather","arg')
  assert.equal(r.text, 'Hello ')
  assert.equal(r.calls.length, 0)
  r = p.push('uments":{"city":"Paris"}}</tool_call> done')
  assert.equal(r.text, ' done')
  assert.equal(r.calls.length, 1)
  assert.equal(r.calls[0].name, 'get_weather')
  assert.deepEqual(r.calls[0].arguments, { city: 'Paris' })
  const end = p.end()
  assert.equal(end.text, '')
  assert.equal(end.dropped, undefined)
})

test('parser handles multiple blocks and surrounding text', () => {
  const p = createToolCallParser()
  const r = p.push('pre <tool_call>{"name":"a","arguments":{}}</tool_call> mid <tool_call>{"name":"b","arguments":{"x":1}}</tool_call> post')
  assert.equal(r.text, 'pre  mid  post')
  assert.deepEqual(r.calls.map((c) => c.name), ['a', 'b'])
})

test('parser holds back tag split mid-token across deltas', () => {
  // Live Quick streams "<" then "tool_call>{...}</tool_call>" as separate deltas.
  const p = createToolCallParser()
  let r = p.push('pre<')
  assert.equal(r.text, 'pre') // '<' held, not flushed
  assert.equal(r.calls.length, 0)
  r = p.push('tool_call>{"name":"f","arguments":{}}</tool_call>post')
  assert.equal(r.text, 'post')
  assert.equal(r.calls.length, 1)
  assert.equal(r.calls[0].name, 'f')
})

test('parser holds back split close tag', () => {
  const p = createToolCallParser()
  p.push('<tool_call>{"name":"f","arguments":{}}</tool_')
  const r = p.push('call>tail')
  assert.equal(r.text, 'tail')
  assert.equal(r.calls.length, 1)
})

test('parser drops unterminated block without leaking it', () => {
  const p = createToolCallParser()
  p.push('text <tool_call>{"name":"x"')
  const end = p.end()
  assert.equal(end.text, '')
  assert.equal(end.dropped, 'unterminated')
})

test('parser tolerates arguments as JSON string and missing args', () => {
  const p = createToolCallParser()
  const r = p.push('<tool_call>{"name":"f","arguments":"{\\"a\\":1}"}</tool_call><tool_call>{"name":"g"}</tool_call>')
  assert.equal(r.calls.length, 2)
  assert.deepEqual(r.calls[0].arguments, { a: 1 })
  assert.deepEqual(r.calls[1].arguments, {})
})

test('parser ignores malformed JSON blocks', () => {
  const p = createToolCallParser()
  const r = p.push('<tool_call>not json</tool_call>ok')
  assert.equal(r.calls.length, 0)
  assert.equal(r.text, 'ok')
  const end = p.end()
  assert.equal(end.dropped, 'invalid') // malformed block counts as dropped
})

test('parser recovers a call wrapped in stray markup (</invoke>, code fence)', () => {
  const p = createToolCallParser()
  const r1 = p.push('<tool_call>{"name":"Bash","arguments":{"command":"echo hi"}}</invoke>\n</tool_call>')
  const r2 = p.push('<tool_call>```json\n{"name":"Read","arguments":{"file_path":"a.txt"}}\n```</tool_call>')
  const end = p.end()
  assert.deepEqual([...r1.calls, ...r2.calls].map((c) => c.name), ['Bash', 'Read'])
  assert.equal(r1.calls[0].arguments.command, 'echo hi')
  assert.equal(end.dropped, undefined)
  // still rejects real garbage
  const q = createToolCallParser()
  q.push('<tool_call>{"name":"Bash","arguments":{"command":"echo</tool_call>')
  assert.equal(q.end().dropped, 'invalid')
})

test('parser repairs raw newlines and tabs inside string values', () => {
  const p = createToolCallParser()
  const r = p.push('<tool_call>{"name":"Write","arguments":{"file_path":"a.py","content":"def f():\n\treturn \\"x\\"\n"}}</tool_call>')
  assert.equal(r.calls.length, 1)
  assert.equal(r.calls[0].arguments.content, 'def f():\n\treturn "x"\n')
  assert.equal(p.end().dropped, undefined)
})

test('sources stripper drops only citation-only blocks outside code; XML <sources> survives', () => {
  const run = (text, step = 1) => {
    const st = createSourcesStripper()
    let out = ''
    for (let i = 0; i < text.length; i += step) out += st.push(text.slice(i, i + step))
    return out + st.end()
  }
  // citation-only blocks go, split across deltas or not
  assert.equal(run('Done.\n<sources> [numeric tool args](mem://d5e9) </sources>\n\nNext <option>'), 'Done.\n\n\nNext <option>')
  assert.equal(run('a<sources>[x](mem://1), [y](https://e.com/p)</sources>b', 3), 'ab')
  // a real XML <sources> element survives
  const maven = '<sources>\n  <source>src/gen/java</source>\n</sources>'
  assert.equal(run(`Add:\n${maven}\n`), `Add:\n${maven}\n`)
  // even a citation-shaped block is kept inside a code fence
  const fenced = '```md\n<sources>[x](mem://1)</sources>\n```\nafter<sources>[y](mem://2)</sources>'
  assert.equal(run(fenced), '```md\n<sources>[x](mem://1)</sources>\n```\nafter')
  // unclosed / oversized blocks pass through verbatim
  assert.equal(run('a <sources>never closed'), 'a <sources>never closed')
  const big = '<sources>' + 'x'.repeat(5000)
  assert.equal(run(big, 97), big)
})

test('end() salvages a complete JSON block truncated before its close tag', () => {
  // The model finished the JSON but token truncation ate </tool_call>.
  // Leading text was already flushed by push(); end() must not drop the call.
  const p = createToolCallParser()
  const r = p.push('answering<tool_call>{"name":"Bash","arguments":{"command":"echo hi"}}')
  assert.equal(r.text, 'answering')
  const end = p.end()
  assert.equal(end.dropped, undefined) // not dropped
  assert.equal(end.calls.length, 1)
  assert.equal(end.calls[0].name, 'Bash')
  assert.equal(end.calls[0].arguments.command, 'echo hi')
})

test('end() still drops a block truncated mid-string', () => {
  // No complete JSON object can be recovered → fall back to dropped.
  const p = createToolCallParser()
  p.push('<tool_call>{"name":"Bash","arguments":{"command":"echo h')
  const end = p.end()
  assert.equal(end.dropped, 'unterminated')
  assert.equal(end.calls.length, 0)
})

test('end() salvages a complete JSON block split across deltas at EOS', () => {
  const p = createToolCallParser()
  let r = p.push('text <tool_call>{"name":"Read"')
  assert.equal(r.calls.length, 0)
  r = p.push(',"arguments":{"file_path":"a.txt"}}')
  assert.equal(r.calls.length, 0) // still no close tag in stream
  const end = p.end()
  assert.equal(end.dropped, undefined)
  assert.equal(end.calls.length, 1)
  assert.equal(end.calls[0].name, 'Read')
  assert.equal(end.calls[0].arguments.file_path, 'a.txt')
})

test('parser ignores literal </tool_call> inside JSON string arguments across arbitrary splits', () => {
  const stream = 'ok <tool_call>{"name":"Write","arguments":{"file_path":"tools.mjs","content":"const CLOSE = \\"</tool_call>\\";\\n// </tool_call>"}}</tool_call> done'
  for (const step of [stream.length, 7, 1]) {
    const p = createToolCallParser()
    let text = ''
    const calls = []
    for (let i = 0; i < stream.length; i += step) {
      const r = p.push(stream.slice(i, i + step))
      text += r.text
      calls.push(...r.calls)
    }
    const end = p.end()
    text += end.text
    calls.push(...end.calls)
    assert.equal(end.dropped, undefined, `step=${step} must not drop`)
    assert.equal(text, 'ok  done', `step=${step} visible text`)
    assert.equal(calls.length, 1, `step=${step} single call`)
    assert.equal(calls[0].name, 'Write')
    assert.equal(calls[0].arguments.content, 'const CLOSE = "</tool_call>";\n// </tool_call>')
  }
})

test('aliasClientTools renames colliding web_search variants and avoids name collisions', () => {
  const tools = normalizeTools([
    { type: 'function', function: { name: 'web_search', description: 'ws' } },
    { type: 'function', function: { name: 'WebSearch', description: 'cc' } },
    { type: 'function', function: { name: 'brave_web_search', description: 'bws' } },
    { type: 'function', function: { name: 'fetch_content', description: 'fc' } },
  ])
  const { tools: mapped, toUpstream, toClient } = aliasClientTools(tools)
  assert.deepEqual(mapped.map((t) => t.name), ['client_web_search', 'client_WebSearch', 'brave_web_search', 'fetch_content'])
  assert.equal(toUpstream.get('web_search'), 'client_web_search')
  assert.equal(toUpstream.get('WebSearch'), 'client_WebSearch')
  assert.equal(toUpstream.has('brave_web_search'), false)
  assert.equal(toClient.get('client_web_search'), 'web_search')
  assert.equal(toClient.get('client_WebSearch'), 'WebSearch')

  // Collision avoidance if client_web_search is already declared
  const colliding = normalizeTools([
    { type: 'function', function: { name: 'web_search' } },
    { type: 'function', function: { name: 'client_web_search' } },
  ])
  const res2 = aliasClientTools(colliding)
  assert.deepEqual(res2.tools.map((t) => t.name), ['client_web_search_2', 'client_web_search'])
  assert.equal(res2.toClient.get('client_web_search_2'), 'web_search')
})

test('aliasToolChoice rewrites named OpenAI and Anthropic tool choices', () => {
  const toUpstream = new Map([['web_search', 'client_web_search']])
  assert.equal(aliasToolChoice('auto', toUpstream), 'auto')
  assert.equal(aliasToolChoice('required', toUpstream), 'required')
  assert.deepEqual(
    aliasToolChoice({ type: 'function', function: { name: 'web_search' } }, toUpstream),
    { type: 'function', function: { name: 'client_web_search' } },
  )
  assert.deepEqual(
    aliasToolChoice({ type: 'tool', name: 'web_search' }, toUpstream),
    { type: 'tool', name: 'client_web_search' },
  )
  assert.deepEqual(
    aliasToolChoice({ type: 'function', function: { name: 'fetch_content' } }, toUpstream),
    { type: 'function', function: { name: 'fetch_content' } },
  )
})

