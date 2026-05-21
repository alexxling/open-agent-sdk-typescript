/**
 * canUseTool denial audit regression tests.
 *
 * Background: SDKResultMessage advertised a `permission_denials` array and
 * HOOK_EVENTS advertised `PermissionRequest` / `PermissionDenied`, but the
 * engine never wrote to the array and never fired the hooks. This file
 * pins down the contract for the fix:
 *
 *   1. result.permission_denials accumulates every `{ behavior: 'deny' }`.
 *   2. PermissionDenied hook is fired with the standard HookInput fields.
 *   3. PermissionRequest hook fires before every canUseTool call (allow OR deny).
 *   4. canUseTool throwing is counted as a denial too (not just deny return).
 *
 * Mirrors the style of tests/structured-output.test.ts: tsx-runnable, MockProvider,
 * self-contained assert helpers. Run: npx tsx tests/permissions-audit.test.ts
 */

import { createAgent, defineTool } from '../src/index.js'
import type {
  LLMProvider,
  CreateMessageParams,
  CreateMessageResponse,
} from '../src/providers/types.js'
import type {
  HookInput,
  SDKMessage,
  ToolDefinition,
} from '../src/index.js'

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
// Mock provider: replays scripted responses turn-by-turn.
// --------------------------------------------------------------------------

interface ScriptedResponse {
  content: CreateMessageResponse['content']
  stopReason?: CreateMessageResponse['stopReason']
}

class MockProvider implements LLMProvider {
  readonly apiType = 'openai-completions' as const
  public calls: CreateMessageParams[] = []
  private script: ScriptedResponse[]

  constructor(script: ScriptedResponse[]) {
    this.script = [...script]
  }

  async createMessage(params: CreateMessageParams): Promise<CreateMessageResponse> {
    this.calls.push({
      ...params,
      messages: structuredClone(params.messages),
    })
    const next = this.script.shift()
    if (!next) {
      throw new Error('MockProvider: script exhausted (engine called more turns than scripted)')
    }
    return {
      content: next.content,
      stopReason: next.stopReason ?? 'end_turn',
      usage: { input_tokens: 10, output_tokens: 20 },
    }
  }
}

/** Bypass the createProvider() factory and swap in a mock. */
function withMockProvider(agent: any, mock: MockProvider): MockProvider {
  agent.provider = mock
  return mock
}

// --------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------

/**
 * Minimal in-process Read tool. The tool body is reachable only in the
 * "allow" test; on the deny / throw paths the engine short-circuits before
 * calling `call`. Kept harmless so a stray invocation can't crash the suite.
 *
 * Matches the defineTool() helper contract: `call` returns `string` (treated
 * as the tool_result content) and `isReadOnly` is a plain boolean.
 */
const fakeReadTool: ToolDefinition = defineTool({
  name: 'Read',
  description: 'Fake Read for tests',
  inputSchema: {
    type: 'object',
    properties: { file_path: { type: 'string' } },
    required: ['file_path'],
  },
  isReadOnly: true,
  call: async () => 'fake-read-output',
})

/**
 * Build a two-turn script: turn 1 emits a tool_use(Read); turn 2 emits a
 * text answer with end_turn. This is enough to drive a full agentic loop
 * through the deny path and out to a final `result` event.
 */
function denyScenarioScript(): ScriptedResponse[] {
  return [
    {
      content: [
        {
          type: 'tool_use',
          id: 'toolu_test_1',
          name: 'Read',
          input: { file_path: './README.md' },
        },
      ],
      stopReason: 'tool_use',
    },
    {
      content: [{ type: 'text', text: 'ok, I cannot read that file.' }],
      stopReason: 'end_turn',
    },
  ]
}

/** Run an agent.query stream to completion, collecting every event. */
async function collect(agent: any, prompt: string): Promise<SDKMessage[]> {
  const events: SDKMessage[] = []
  for await (const ev of agent.query(prompt)) {
    events.push(ev)
  }
  return events
}

function findResult(events: SDKMessage[]): any {
  return events.find((e: any) => e.type === 'result')
}

function findToolResults(events: SDKMessage[]): any[] {
  return events.filter((e: any) => e.type === 'tool_result')
}

// --------------------------------------------------------------------------
// Test 1: deny populates result.permission_denials
// --------------------------------------------------------------------------

async function test_denyPopulatesAudit() {
  section('Test 1: result.permission_denials accumulates canUseTool denials')

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [fakeReadTool],
    canUseTool: async () => ({
      behavior: 'deny',
      message: 'no read in test',
    }),
  })
  withMockProvider(agent, new MockProvider(denyScenarioScript()))

  const events = await collect(agent, 'please read README.md')
  const result = findResult(events)

  assert(result !== undefined, 'final result event is emitted')
  assert(
    Array.isArray(result?.permission_denials) && result.permission_denials.length >= 1,
    'result.permission_denials has at least one entry',
  )
  assertEqual(
    result?.permission_denials?.[0]?.tool,
    'Read',
    'first denial records the tool name',
  )
  assert(
    typeof result?.permission_denials?.[0]?.reason === 'string' &&
      result.permission_denials[0].reason.includes('no read in test'),
    'first denial reason includes the canUseTool message',
  )

  // Existing behaviour preserved: the denied call is still surfaced as an
  // is_error tool_result so the model can self-correct.
  const toolResults = findToolResults(events)
  assert(
    toolResults.some(
      (t: any) =>
        t.result.tool_name === 'Read' && t.result.output.includes('no read in test'),
    ),
    'a tool_result event still carries the denial message back (LLM-facing)',
  )
}

// --------------------------------------------------------------------------
// Test 2: PermissionDenied hook fires with expected HookInput fields
// --------------------------------------------------------------------------

async function test_permissionDeniedHookFires() {
  section('Test 2: PermissionDenied hook fires with toolName/toolInput/toolUseId/error')

  const deniedCalls: HookInput[] = []

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [fakeReadTool],
    canUseTool: async () => ({ behavior: 'deny', message: 'nope' }),
    hooks: {
      PermissionDenied: [
        {
          hooks: [
            async (input: any) => {
              deniedCalls.push(input as HookInput)
            },
          ],
        },
      ],
    },
  })
  withMockProvider(agent, new MockProvider(denyScenarioScript()))

  await collect(agent, 'please read README.md')

  assert(deniedCalls.length >= 1, 'PermissionDenied hook was invoked at least once')
  const first = deniedCalls[0]
  assertEqual(first?.toolName, 'Read', 'hook input.toolName === "Read"')
  assert(
    first?.toolInput !== undefined &&
      typeof (first.toolInput as any)?.file_path === 'string',
    'hook input.toolInput carries the original tool input',
  )
  assertEqual(first?.toolUseId, 'toolu_test_1', 'hook input.toolUseId matches the tool_use id')
  assert(
    typeof first?.error === 'string' && first.error.includes('nope'),
    'hook input.error contains the deny reason',
  )
}

// --------------------------------------------------------------------------
// Test 3: PermissionRequest fires before every canUseTool call (incl. allow)
// --------------------------------------------------------------------------

async function test_permissionRequestFiresOnAllow() {
  section('Test 3: PermissionRequest fires before each canUseTool call (allow path)')

  const requestCalls: HookInput[] = []
  let canUseToolCalls = 0

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [fakeReadTool],
    canUseTool: async () => {
      canUseToolCalls++
      return { behavior: 'allow' }
    },
    hooks: {
      PermissionRequest: [
        {
          hooks: [
            async (input: any) => {
              requestCalls.push(input as HookInput)
            },
          ],
        },
      ],
    },
  })
  withMockProvider(agent, new MockProvider(denyScenarioScript()))

  await collect(agent, 'please read README.md')

  assertEqual(canUseToolCalls, 1, 'canUseTool was invoked exactly once (one tool_use in the script)')
  assertEqual(requestCalls.length, 1, 'PermissionRequest hook fired exactly once')
  assertEqual(requestCalls[0]?.toolName, 'Read', 'PermissionRequest carries toolName')
  assertEqual(
    requestCalls[0]?.toolUseId,
    'toolu_test_1',
    'PermissionRequest carries toolUseId',
  )
}

// --------------------------------------------------------------------------
// Test 4: canUseTool throwing is counted as a denial
// --------------------------------------------------------------------------

async function test_throwCountsAsDenial() {
  section('Test 4: canUseTool throwing is recorded as a permission denial')

  const deniedCalls: HookInput[] = []

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [fakeReadTool],
    canUseTool: async () => {
      throw new Error('boom')
    },
    hooks: {
      PermissionDenied: [
        {
          hooks: [
            async (input: any) => {
              deniedCalls.push(input as HookInput)
            },
          ],
        },
      ],
    },
  })
  withMockProvider(agent, new MockProvider(denyScenarioScript()))

  const events = await collect(agent, 'please read README.md')
  const result = findResult(events)

  assert(
    Array.isArray(result?.permission_denials) && result.permission_denials.length >= 1,
    'thrown error produces a permission_denials entry',
  )
  const reason = result?.permission_denials?.[0]?.reason as string | undefined
  assert(
    typeof reason === 'string' &&
      reason.includes('Permission check error') &&
      reason.includes('boom'),
    'denial reason mentions both "Permission check error" and the underlying message',
  )

  // The pre-existing fail-soft behaviour (is_error tool_result with the same
  // text) MUST be preserved so the model still sees the failure.
  const toolResults = findToolResults(events)
  assert(
    toolResults.some(
      (t: any) =>
        t.result.tool_name === 'Read' &&
        typeof t.result.output === 'string' &&
        t.result.output.includes('Permission check error') &&
        t.result.output.includes('boom'),
    ),
    'a tool_result with the error message is still emitted to the LLM',
  )

  // And the PermissionDenied hook still fires on the throw path.
  assert(deniedCalls.length >= 1, 'PermissionDenied hook fires when canUseTool throws')
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

async function main() {
  console.log('=== canUseTool denial audit test suite ===')
  await test_denyPopulatesAudit()
  await test_permissionDeniedHookFires()
  await test_permissionRequestFiresOnAllow()
  await test_throwCountsAsDenial()

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
