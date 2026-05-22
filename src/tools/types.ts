/**
 * Tool interface and helper utilities
 */

import type { ToolDefinition, ToolInputSchema, ToolContext, ToolResult } from '../types.js'

/**
 * Helper to create a tool definition with sensible defaults.
 *
 * `prompt` semantics (consumed by engine.ts > buildSystemPrompt + toProviderTool):
 *   - omitted          → no `prompt` field on the resulting ToolDefinition; the
 *                        engine uses `description` as-is (back-compat: matches
 *                        every built-in tool that never set this field).
 *   - string           → wrapped as `async () => string` so the engine can
 *                        append it after `description` on every API turn.
 *   - (ctx) => string  → invoked per turn with a real ToolContext (cwd / model /
 *                        apiType / provider), letting tools tell the LLM things
 *                        like "current cwd is …" without hardcoding at define time.
 *
 * NOTE: prior versions fell back to `description` when `prompt` was missing,
 * which had no effect (no consumer) but would now double-inject the description.
 * We therefore only emit a `prompt` callback when the caller explicitly supplied one.
 */
export function defineTool(config: {
  name: string
  description: string
  inputSchema: ToolInputSchema
  call: (input: any, context: ToolContext) => Promise<string | { data: string; is_error?: boolean }>
  isReadOnly?: boolean
  isConcurrencySafe?: boolean
  prompt?: string | ((context: ToolContext) => Promise<string>)
}): ToolDefinition {
  const promptCallback: ToolDefinition['prompt'] | undefined =
    typeof config.prompt === 'function'
      ? config.prompt
      : typeof config.prompt === 'string'
        ? async () => config.prompt as string
        : undefined

  return {
    name: config.name,
    description: config.description,
    inputSchema: config.inputSchema,
    isReadOnly: () => config.isReadOnly ?? false,
    // Only emit isConcurrencySafe when the caller explicitly opted in or out.
    // The engine's partition logic (executeTools) treats a missing/undefined
    // hint as "compatible default = parallelize alongside other read-only
    // tools", matching how every built-in tool predates this hint. Synthesizing
    // a `() => false` default here (the previous behaviour) silently opted out
    // every defineTool() user from the concurrent bucket — see
    // tests/concurrency-safe.test.ts for the regression that pinned this.
    ...(config.isConcurrencySafe !== undefined
      ? { isConcurrencySafe: () => config.isConcurrencySafe as boolean }
      : {}),
    isEnabled: () => true,
    ...(promptCallback ? { prompt: promptCallback } : {}),
    async call(input: any, context: ToolContext): Promise<ToolResult> {
      try {
        const result = await config.call(input, context)
        const output = typeof result === 'string' ? result : result.data
        const isError = typeof result === 'object' && result.is_error
        return {
          type: 'tool_result',
          tool_use_id: '', // filled by engine
          content: output,
          is_error: isError || false,
        }
      } catch (err: any) {
        return {
          type: 'tool_result',
          tool_use_id: '',
          content: `Error: ${err.message}`,
          is_error: true,
        }
      }
    },
  }
}

/**
 * Convert a ToolDefinition to API-compatible tool format.
 * Returns the normalized tool format used by providers.
 */
export function toApiTool(tool: ToolDefinition): {
  name: string
  description: string
  input_schema: ToolInputSchema
} {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  }
}
