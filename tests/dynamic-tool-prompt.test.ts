/**
 * Dynamic `tool.prompt(ctx)` injection regression tests.
 *
 * Background: ToolDefinition has long declared an optional
 *   prompt?: (context: ToolContext) => Promise<string>
 * field, and `defineTool({ prompt })` faithfully wrapped it onto the returned
 * ToolDefinition — but the engine never read it. So a user-supplied dynamic
 * description was silently dropped and the LLM only ever saw the static
 * `description`. This file pins down the contract for the fix:
 *
 *   1. When a tool exposes `prompt(ctx)`, its return value is appended after
 *      `description` in the system-prompt tool listing.
 *   2. The same composed string is forwarded as the tool's `description` field
 *      in the provider tools array (what the LLM actually receives in the API
 *      `tools` parameter), so the static and dynamic surfaces agree.
 *   3. `prompt(ctx)` receives a real ToolContext (cwd / model / apiType /
 *      provider) so callers can render runtime-specific instructions.
 *   4. Tools without a `prompt` callback (the historical default for every
 *      built-in tool) keep emitting their static description verbatim.
 *   5. A throwing prompt() must NEVER break the agent loop — the engine falls
 *      back to the static description.
 *
 * Mirrors the style of tests/permissions-audit.test.ts: tsx-runnable, MockProvider,
 * self-contained assert helpers. Run: npx tsx tests/dynamic-tool-prompt.test.ts
 */

import { createAgent, defineTool } from '../src/index.js'
import type {
  LLMProvider,
  CreateMessageParams,
  CreateMessageResponse,
} from '../src/providers/types.js'
import type { SDKMessage, ToolContext } from '../src/index.js'

// --------------------------------------------------------------------------
// Tiny assertion helper (kept in-file to avoid pulling in a test runner)
// --------------------------------------------------------------------------

let passed = 0
let failed = 0
const failures: string[] = []

function assert(cond: unknown, msg: string): void {
  if (cond) {
    passed++
    console.log(`  ok    ${msg}`)
  } else {
    failed++
    failures.push(msg)
    console.log(`  FAIL  ${msg}`)
  }
}

function assertEqual(actual: unknown, expected: unknown, msg: string): void {
  const a = JSON.stringify(actual)
  const e = JSON.stringify(expected)
  if (a === e) {
    passed++
    console.log(`  ok    ${msg}`)
  } else {
    failed++
    failures.push(`${msg}\n        expected: ${e}\n        actual:   ${a}`)
    console.log(`  FAIL  ${msg}`)
    console.log(`        expected: ${e}`)
    console.log(`        actual:   ${a}`)
  }
}

function section(title: string): void {
  console.log(`\n--- ${title} ---`)
}

// --------------------------------------------------------------------------
// Mock provider: single-turn "text reply, end_turn" script. Enough to drive
// the engine through buildSystemPrompt + toProviderTool and out to a result.
// We capture every CreateMessageParams so the test can inspect both the
// `system` string and the `tools` array.
// --------------------------------------------------------------------------

class MockProvider implements LLMProvider {
  readonly apiType = 'openai-completions' as const
  public calls: CreateMessageParams[] = []

  async createMessage(params: CreateMessageParams): Promise<CreateMessageResponse> {
    this.calls.push({
      ...params,
      messages: structuredClone(params.messages),
    })
    return {
      content: [{ type: 'text', text: 'noop' }],
      stopReason: 'end_turn',
      usage: { input_tokens: 1, output_tokens: 1 },
    }
  }
}

function withMockProvider(agent: any, mock: MockProvider): MockProvider {
  agent.provider = mock
  return mock
}

async function collect(agent: any, prompt: string): Promise<SDKMessage[]> {
  const events: SDKMessage[] = []
  for await (const ev of agent.query(prompt)) {
    events.push(ev)
  }
  return events
}

function findProviderTool(call: CreateMessageParams, name: string) {
  return call.tools?.find((t: any) => t.name === name)
}

// --------------------------------------------------------------------------
// Test 1: string-form `prompt` is appended to description in both surfaces
// --------------------------------------------------------------------------

async function test_stringPromptIsAppended() {
  section('Test 1: string-form `prompt` is appended after description')

  const STATIC_DESC = 'Lookup current time.'
  const DYNAMIC_HINT = 'CUSTOM_PROMPT_MARKER_v1 — always prefer ISO 8601.'

  const tool = defineTool({
    name: 'now_tool',
    description: STATIC_DESC,
    inputSchema: { type: 'object', properties: {}, required: [] },
    isReadOnly: true,
    prompt: DYNAMIC_HINT,
    call: async () => 'unused',
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [tool],
  })
  const mock = withMockProvider(agent, new MockProvider())

  await collect(agent, 'what time is it')

  assertEqual(mock.calls.length, 1, 'provider receives exactly one createMessage call')
  const call = mock.calls[0]

  assert(
    typeof call.system === 'string' && call.system.includes(DYNAMIC_HINT),
    'system prompt contains the dynamic hint (string-form prompt was injected)',
  )
  assert(
    typeof call.system === 'string' && call.system.includes(STATIC_DESC),
    'system prompt still contains the static description (not replaced, appended)',
  )

  const provTool = findProviderTool(call, 'now_tool')
  assert(provTool !== undefined, 'provider tools array contains the registered tool')
  assert(
    typeof provTool?.description === 'string' &&
      provTool.description.includes(DYNAMIC_HINT) &&
      provTool.description.includes(STATIC_DESC),
    'provider tool.description carries BOTH static + dynamic (system & tools agree)',
  )
}

// --------------------------------------------------------------------------
// Test 2: function-form prompt(ctx) is awaited and receives the live ToolContext
// --------------------------------------------------------------------------

async function test_functionPromptReceivesToolContext() {
  section('Test 2: function-form prompt(ctx) receives live ToolContext')

  const seenCtx: ToolContext[] = []
  const tool = defineTool({
    name: 'cwd_aware_tool',
    description: 'Knows your project layout.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    isReadOnly: true,
    prompt: async (ctx) => {
      seenCtx.push(ctx)
      return `DYN_CWD=${ctx.cwd}|DYN_MODEL=${ctx.model}|DYN_API=${ctx.apiType}`
    },
    call: async () => 'unused',
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    cwd: '/tmp/agent-test-cwd',
    tools: [tool],
  })
  const mock = withMockProvider(agent, new MockProvider())

  await collect(agent, 'noop')

  assert(seenCtx.length >= 1, 'prompt(ctx) callback was actually invoked')
  const ctx = seenCtx[0]
  assertEqual(ctx.cwd, '/tmp/agent-test-cwd', 'ctx.cwd matches Agent option cwd')
  assertEqual(ctx.model, 'gpt-4o', 'ctx.model matches Agent option model')
  assertEqual(
    ctx.apiType,
    'openai-completions',
    'ctx.apiType matches the configured provider apiType',
  )

  const call = mock.calls[0]
  assert(
    typeof call.system === 'string' &&
      call.system.includes('DYN_CWD=/tmp/agent-test-cwd') &&
      call.system.includes('DYN_MODEL=gpt-4o') &&
      call.system.includes('DYN_API=openai-completions'),
    'rendered dynamic string is injected verbatim into system prompt',
  )
  const provTool = findProviderTool(call, 'cwd_aware_tool')
  assert(
    typeof provTool?.description === 'string' &&
      provTool.description.includes('DYN_CWD=/tmp/agent-test-cwd'),
    'rendered dynamic string is also injected into provider tool.description',
  )
}

// --------------------------------------------------------------------------
// Test 3: tools without a `prompt` callback are unaffected (back-compat)
// --------------------------------------------------------------------------

async function test_noPromptCallbackIsBackCompat() {
  section('Test 3: tools without prompt() keep emitting static description verbatim')

  const STATIC_DESC = 'No-prompt fallback tool.'
  const tool = defineTool({
    name: 'plain_tool',
    description: STATIC_DESC,
    inputSchema: { type: 'object', properties: {}, required: [] },
    isReadOnly: true,
    // no `prompt` field — same shape every built-in tool has historically used
    call: async () => 'unused',
  })

  // Spec assertion: defineTool MUST NOT synthesize a prompt callback when the
  // caller omits it. Otherwise we'd silently double-inject the description.
  assertEqual(
    typeof tool.prompt,
    'undefined',
    'defineTool({...}) without prompt does not populate a prompt callback',
  )

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [tool],
  })
  const mock = withMockProvider(agent, new MockProvider())

  await collect(agent, 'noop')
  const call = mock.calls[0]
  const provTool = findProviderTool(call, 'plain_tool')

  assertEqual(
    provTool?.description,
    STATIC_DESC,
    'provider tool.description equals the static description (no suffix appended)',
  )
  // Count occurrences of the static description in the system prompt — should
  // be exactly 1 (the "# Available Tools" line). Anything ≥ 2 would mean we
  // duplicated the description via a default-fallback prompt() somewhere.
  const sys = (call.system as string) || ''
  const occurrences = sys.split(STATIC_DESC).length - 1
  assertEqual(
    occurrences,
    1,
    'static description appears exactly once in system prompt (no accidental duplication)',
  )
}

// --------------------------------------------------------------------------
// Test 4: a throwing prompt() falls back to the static description, never crashes
// --------------------------------------------------------------------------

async function test_throwingPromptFallsBack() {
  section('Test 4: throwing prompt(ctx) falls back to static description without breaking the loop')

  const STATIC_DESC = 'I survive crashes.'
  let callCount = 0
  const tool = defineTool({
    name: 'crashy_prompt_tool',
    description: STATIC_DESC,
    inputSchema: { type: 'object', properties: {}, required: [] },
    isReadOnly: true,
    prompt: async () => {
      callCount++
      throw new Error('prompt() boom')
    },
    call: async () => 'unused',
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [tool],
  })
  const mock = withMockProvider(agent, new MockProvider())

  // Should NOT throw — the engine is required to swallow prompt() errors.
  let crashed = false
  try {
    await collect(agent, 'noop')
  } catch {
    crashed = true
  }
  assert(!crashed, 'agent loop survives a prompt() that throws')

  // prompt() is invoked at least once: from buildSystemPrompt OR toProviderTool
  // (both call resolveToolDescription). 1 or 2 calls are both spec-compliant.
  assert(callCount >= 1, 'throwing prompt() callback was actually invoked')

  const call = mock.calls[0]
  const provTool = findProviderTool(call, 'crashy_prompt_tool')
  assertEqual(
    provTool?.description,
    STATIC_DESC,
    'provider tool.description falls back to static description after prompt() throw',
  )
  const sys = (call.system as string) || ''
  assert(
    sys.includes(STATIC_DESC),
    'system prompt still contains the static description after prompt() throw',
  )
  assert(
    !sys.includes('boom'),
    'error message from a throwing prompt() does not leak into the system prompt',
  )
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

async function main() {
  console.log('=== dynamic tool prompt injection test suite ===')
  await test_stringPromptIsAppended()
  await test_functionPromptReceivesToolContext()
  await test_noPromptCallbackIsBackCompat()
  await test_throwingPromptFallsBack()

  console.log(`\n=== ${passed} passed, ${failed} failed ===`)
  if (failed > 0) {
    console.log('\nFailures:')
    for (const f of failures) console.log(`  - ${f}`)
    process.exit(1)
  }
}

main().catch((err) => {
  console.error('test runner crashed:', err)
  process.exit(1)
})
