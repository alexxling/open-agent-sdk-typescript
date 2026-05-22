/**
 * `tool.isConcurrencySafe()` partition regression tests.
 *
 * Background: ToolDefinition has long declared an optional
 *   isConcurrencySafe?: () => boolean
 * field, and `defineTool({ isConcurrencySafe })` faithfully wrapped it onto
 * the returned ToolDefinition — but the engine's executeTools() partition only
 * looked at `isReadOnly()`. So a tool tagged "read-only but NOT safe to call
 * in parallel" (rate-limited API, stateful cursor, order-sensitive scrape)
 * would still be batched into the concurrent bucket. This file pins down the
 * contract for the fix:
 *
 *   1. Two read-only-safe tools called in the same turn run CONCURRENTLY
 *      (their call windows overlap). [historical behaviour preserved]
 *   2. A read-only-but-unsafe tool (isReadOnly:true + isConcurrencySafe:false)
 *      is pushed to the serial bucket: it does NOT start until every
 *      read-only-safe call in the same turn has finished.
 *   3. Tools that omit isConcurrencySafe entirely keep parallelizing — this is
 *      the back-compat path that every built-in tool relies on.
 *   4. A non-read-only tool (mutation) still runs serially regardless of the
 *      isConcurrencySafe value — the read-only gate is a hard prerequisite for
 *      the concurrent bucket.
 *
 * Mirrors the style of tests/permissions-audit.test.ts: tsx-runnable, MockProvider,
 * self-contained assert helpers. Run: npx tsx tests/concurrency-safe.test.ts
 */

import { createAgent, defineTool } from '../src/index.js'
import type {
  LLMProvider,
  CreateMessageParams,
  CreateMessageResponse,
} from '../src/providers/types.js'
import type { SDKMessage } from '../src/index.js'

// --------------------------------------------------------------------------
// Tiny assertion helper
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

function section(title: string): void {
  console.log(`\n--- ${title} ---`)
}

// --------------------------------------------------------------------------
// Mock provider: scripted, just like tests/permissions-audit.test.ts
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
      throw new Error('MockProvider: script exhausted')
    }
    return {
      content: next.content,
      stopReason: next.stopReason ?? 'end_turn',
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

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/**
 * Make a tool whose call() pushes timestamped enter/exit markers into a shared
 * log. The relative ordering of markers between tools is the partition oracle.
 */
function makeTimingTool(opts: {
  name: string
  isReadOnly: boolean
  isConcurrencySafe?: boolean
  delayMs: number
  log: string[]
}) {
  const config = {
    name: opts.name,
    description: `${opts.name} (delay=${opts.delayMs}ms)`,
    inputSchema: { type: 'object' as const, properties: {}, required: [] as string[] },
    isReadOnly: opts.isReadOnly,
    call: async () => {
      opts.log.push(`${opts.name}:enter`)
      await sleep(opts.delayMs)
      opts.log.push(`${opts.name}:exit`)
      return 'ok'
    },
    ...(opts.isConcurrencySafe !== undefined
      ? { isConcurrencySafe: opts.isConcurrencySafe }
      : {}),
  }
  return defineTool(config)
}

/** Build a 2-turn script: turn 1 emits the given tool_use blocks; turn 2 ends. */
function turnWithToolUses(blocks: Array<{ id: string; name: string }>): ScriptedResponse[] {
  return [
    {
      content: blocks.map((b) => ({
        type: 'tool_use' as const,
        id: b.id,
        name: b.name,
        input: {},
      })),
      stopReason: 'tool_use',
    },
    {
      content: [{ type: 'text', text: 'done' }],
      stopReason: 'end_turn',
    },
  ]
}

/**
 * Pure log-ordering predicate.
 *
 * "Two windows overlap" === `b:enter` happens after `a:enter` but before
 * `a:exit`. Order-insensitive in `a`/`b`: we test both directions.
 */
function windowsOverlap(log: string[], a: string, b: string): boolean {
  const aEnter = log.indexOf(`${a}:enter`)
  const aExit = log.indexOf(`${a}:exit`)
  const bEnter = log.indexOf(`${b}:enter`)
  const bExit = log.indexOf(`${b}:exit`)
  if (aEnter < 0 || aExit < 0 || bEnter < 0 || bExit < 0) return false
  return (
    (bEnter > aEnter && bEnter < aExit) ||
    (aEnter > bEnter && aEnter < bExit)
  )
}

/** `b:enter` strictly after `a:exit` — proves serialization. */
function strictlyAfter(log: string[], a: string, b: string): boolean {
  const aExit = log.indexOf(`${a}:exit`)
  const bEnter = log.indexOf(`${b}:enter`)
  if (aExit < 0 || bEnter < 0) return false
  return bEnter > aExit
}

// --------------------------------------------------------------------------
// Test 1: two safe read-only tools run concurrently (preserved behaviour)
// --------------------------------------------------------------------------

async function test_twoSafeReadOnlyRunConcurrently() {
  section('Test 1: isReadOnly:true + isConcurrencySafe:true (or omitted) parallelize')

  const log: string[] = []
  // One tool sets the hint explicitly; the other omits it (back-compat path).
  const safeA = makeTimingTool({
    name: 'safe_a',
    isReadOnly: true,
    isConcurrencySafe: true,
    delayMs: 60,
    log,
  })
  const safeBOmitted = makeTimingTool({
    name: 'safe_b_omitted',
    isReadOnly: true,
    // isConcurrencySafe intentionally omitted
    delayMs: 60,
    log,
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [safeA, safeBOmitted],
  })
  withMockProvider(
    agent,
    new MockProvider(
      turnWithToolUses([
        { id: 'tu_a', name: 'safe_a' },
        { id: 'tu_b', name: 'safe_b_omitted' },
      ]),
    ),
  )

  await collect(agent, 'noop')

  assert(
    windowsOverlap(log, 'safe_a', 'safe_b_omitted'),
    'two safe read-only call windows overlap (ran in the concurrent bucket)',
  )
}

// --------------------------------------------------------------------------
// Test 2: a read-only-but-unsafe tool is opted OUT of the concurrent bucket
// --------------------------------------------------------------------------

async function test_unsafeReadOnlyIsSerialized() {
  section('Test 2: isReadOnly:true + isConcurrencySafe:false runs AFTER the concurrent batch')

  const log: string[] = []
  // Concurrent bucket member: slow on purpose.
  const safe = makeTimingTool({
    name: 'safe_slow',
    isReadOnly: true,
    isConcurrencySafe: true,
    delayMs: 80,
    log,
  })
  // Serial bucket member: fast — if the engine treated it as concurrent it
  // would finish FIRST. The serial-bucket gate forces it to wait.
  const unsafe = makeTimingTool({
    name: 'unsafe_fast',
    isReadOnly: true,
    isConcurrencySafe: false,
    delayMs: 10,
    log,
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [safe, unsafe],
  })
  withMockProvider(
    agent,
    new MockProvider(
      turnWithToolUses([
        { id: 'tu_safe', name: 'safe_slow' },
        { id: 'tu_unsafe', name: 'unsafe_fast' },
      ]),
    ),
  )

  await collect(agent, 'noop')

  assert(
    strictlyAfter(log, 'safe_slow', 'unsafe_fast'),
    'unsafe_fast:enter occurs after safe_slow:exit (serial after concurrent batch drained)',
  )
  assert(
    !windowsOverlap(log, 'safe_slow', 'unsafe_fast'),
    'safe_slow and unsafe_fast call windows do NOT overlap',
  )
}

// --------------------------------------------------------------------------
// Test 3: a non-read-only tool is always serial regardless of isConcurrencySafe
// --------------------------------------------------------------------------

async function test_mutationStaysSerial() {
  section('Test 3: isReadOnly:false stays serial even when isConcurrencySafe:true')

  const log: string[] = []
  const safe = makeTimingTool({
    name: 'safe_slow',
    isReadOnly: true,
    isConcurrencySafe: true,
    delayMs: 80,
    log,
  })
  // Mutation tool but advertises concurrency-safe; the read-only gate must
  // still keep it out of the concurrent bucket.
  const mutationButSafeHint = makeTimingTool({
    name: 'mutation_fast',
    isReadOnly: false,
    isConcurrencySafe: true,
    delayMs: 10,
    log,
  })

  const agent = createAgent({
    apiType: 'openai-completions',
    apiKey: 'mock',
    baseURL: 'https://example.invalid',
    model: 'gpt-4o',
    tools: [safe, mutationButSafeHint],
  })
  withMockProvider(
    agent,
    new MockProvider(
      turnWithToolUses([
        { id: 'tu_safe', name: 'safe_slow' },
        { id: 'tu_mut', name: 'mutation_fast' },
      ]),
    ),
  )

  await collect(agent, 'noop')

  assert(
    strictlyAfter(log, 'safe_slow', 'mutation_fast'),
    'mutation_fast:enter occurs after safe_slow:exit (mutation stays serial)',
  )
  assert(
    !windowsOverlap(log, 'safe_slow', 'mutation_fast'),
    'safe_slow and mutation_fast windows do NOT overlap',
  )
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

async function main() {
  console.log('=== isConcurrencySafe partition test suite ===')
  await test_twoSafeReadOnlyRunConcurrently()
  await test_unsafeReadOnlyIsSerialized()
  await test_mutationStaysSerial()

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
