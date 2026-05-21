/**
 * `system.init` MCP / permission regression tests.
 *
 * Background: the engine used to hard-code
 *   mcp_servers: [],
 *   permission_mode: 'bypassPermissions',
 * on every `system.init` event, ignoring whatever the Agent layer had actually
 * wired up. This file pins down the contract for the fix:
 *
 *   1. With NO mcpServers configured, init.mcp_servers stays []  (back-compat).
 *   2. With an in-process SDK MCP server, init.mcp_servers carries
 *      `{ name, status: 'connected' }` and init.tools includes the
 *      mcp__<name>__<tool> entries (so allowedTools can target them).
 *   3. init.permission_mode reflects the caller-supplied AgentOptions.permissionMode
 *      (e.g. 'default'), not a hardcoded literal.
 *   4. With permissionMode omitted, init.permission_mode falls back to
 *      'bypassPermissions' (historical default preserved).
 *
 * Mirrors the style of tests/permissions-audit.test.ts: tsx-runnable, MockProvider,
 * self-contained assert helpers. Run: npx tsx tests/mcp-init.test.ts
 */

import { createAgent, createSdkMcpServer, tool } from '../src/index.js'
import { z } from 'zod'
import type {
  LLMProvider,
  CreateMessageParams,
  CreateMessageResponse,
} from '../src/providers/types.js'
import type { SDKMessage } from '../src/index.js'

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
// Mock provider: a single-turn "text reply, end_turn" script. Enough to drive
// the engine through the init event and out to a `result`.
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

function findInit(events: SDKMessage[]): any {
  return events.find((e: any) => e.type === 'system' && e.subtype === 'init')
}

// --------------------------------------------------------------------------
// Fixture: a tiny in-process MCP server with one tool.
// --------------------------------------------------------------------------

function buildShopServer() {
  const getWeather = tool(
    'get_weather',
    'Get weather (mock)',
    { city: z.string() },
    async () => ({ content: [{ type: 'text', text: 'sunny' }] }),
    { annotations: { readOnlyHint: true } },
  )
  return createSdkMcpServer({ name: 'shop', tools: [getWeather] })
}

// --------------------------------------------------------------------------
// Test 1: no mcpServers configured → init.mcp_servers is [] (back-compat).
// --------------------------------------------------------------------------

async function test_noMcpServersStillEmptyArray() {
  section('Test 1: no mcpServers configured → init.mcp_servers is []')

  const agent = createAgent({
    apiType: 'openai-completions',
    model: 'mock-model',
    apiKey: 'mock-key',
    tools: [],
    permissionMode: 'bypassPermissions',
  })
  withMockProvider(agent, new MockProvider())

  const events = await collect(agent, 'hi')
  const init = findInit(events)
  assert(init, 'init event was emitted')
  assertEqual(init.mcp_servers, [], 'init.mcp_servers === []')
}

// --------------------------------------------------------------------------
// Test 2: in-process SDK MCP server → init.mcp_servers carries it as
//         { name, status: 'connected' } and init.tools includes mcp__shop__*.
// --------------------------------------------------------------------------

async function test_sdkMcpServerSurfaced() {
  section('Test 2: in-process SDK MCP server is surfaced on init')

  const shop = buildShopServer()
  const agent = createAgent({
    apiType: 'openai-completions',
    model: 'mock-model',
    apiKey: 'mock-key',
    tools: [],
    mcpServers: { shop },
    permissionMode: 'bypassPermissions',
  })
  withMockProvider(agent, new MockProvider())

  const events = await collect(agent, 'hi')
  const init = findInit(events)
  assert(init, 'init event was emitted')
  assertEqual(
    init.mcp_servers,
    [{ name: 'shop', status: 'connected' }],
    'init.mcp_servers carries the in-process server with status="connected"',
  )
  assert(
    Array.isArray(init.tools) && init.tools.includes('mcp__shop__get_weather'),
    'init.tools contains the mcp__shop__get_weather entry',
  )
}

// --------------------------------------------------------------------------
// Test 3: permissionMode='default' is reflected on init.
// --------------------------------------------------------------------------

async function test_permissionModeReflected() {
  section("Test 3: permissionMode='default' surfaced on init")

  const agent = createAgent({
    apiType: 'openai-completions',
    model: 'mock-model',
    apiKey: 'mock-key',
    tools: [],
    permissionMode: 'default',
  })
  withMockProvider(agent, new MockProvider())

  const events = await collect(agent, 'hi')
  const init = findInit(events)
  assertEqual(init.permission_mode, 'default', "init.permission_mode === 'default'")
}

// --------------------------------------------------------------------------
// Test 4: permissionMode omitted → defaults to 'bypassPermissions'.
// --------------------------------------------------------------------------

async function test_permissionModeDefaultFallback() {
  section('Test 4: permissionMode omitted → init falls back to bypassPermissions')

  const agent = createAgent({
    apiType: 'openai-completions',
    model: 'mock-model',
    apiKey: 'mock-key',
    tools: [],
    // no permissionMode set
  })
  withMockProvider(agent, new MockProvider())

  const events = await collect(agent, 'hi')
  const init = findInit(events)
  assertEqual(
    init.permission_mode,
    'bypassPermissions',
    "init.permission_mode defaults to 'bypassPermissions' for back-compat",
  )
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

async function main() {
  console.log('=== system.init MCP / permission regression suite ===')
  await test_noMcpServersStillEmptyArray()
  await test_sdkMcpServerSurfaced()
  await test_permissionModeReflected()
  await test_permissionModeDefaultFallback()

  console.log(`\n=== ${passed} passed, ${failed} failed ===`)
  if (failed > 0) {
    console.log('\nFailures:')
    for (const f of failures) console.log(`  - ${f}`)
    process.exit(1)
  }
}

void main()
