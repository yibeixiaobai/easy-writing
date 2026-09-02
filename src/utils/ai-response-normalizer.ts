/**
 * OpenAI-compatible 响应归一化。
 *
 * 不同供应商/中转服务虽然都声称兼容 /chat/completions，实际返回的 content
 * 可能是字符串、content part 数组，或者附带 reasoning_content。统一在这里提取
 * 用户真正应该看到的文本；reasoning 只作为诊断信息，不直接写进正文。
 */

export interface NormalizedTextResult {
  text: string
  reasoningOnly: boolean
}

const asNonEmptyString = (value: unknown): string =>
  typeof value === 'string' ? value : ''

/** 兼容 OpenAI content string / content parts / 常见代理自定义 part */
export const extractTextContent = (value: unknown): string => {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''

  return value
    .map(part => {
      if (typeof part === 'string') return part
      if (!part || typeof part !== 'object') return ''
      const item = part as Record<string, unknown>
      if (typeof item.text === 'string') return item.text
      if (typeof item.content === 'string') return item.content
      return ''
    })
    .join('')
}

/** 非流式 chat/completions 响应统一取文本 */
export const normalizeChatCompletionResponse = (body: unknown): NormalizedTextResult => {
  if (!body || typeof body !== 'object') return { text: '', reasoningOnly: false }
  const root = body as Record<string, unknown>
  const choices = Array.isArray(root.choices) ? root.choices : []
  const first = choices[0]
  if (!first || typeof first !== 'object') return { text: '', reasoningOnly: false }

  const choice = first as Record<string, unknown>
  const message =
    choice.message && typeof choice.message === 'object'
      ? (choice.message as Record<string, unknown>)
      : null

  const messageText = extractTextContent(message?.content)
  if (messageText.trim()) return { text: messageText, reasoningOnly: false }

  // 少数兼容层把正文直接放在 choice.text
  const choiceText = asNonEmptyString(choice.text)
  if (choiceText.trim()) return { text: choiceText, reasoningOnly: false }

  const reasoning = extractTextContent(message?.reasoning_content)
  if (reasoning.trim()) return { text: '', reasoningOnly: true }

  return { text: '', reasoningOnly: false }
}

/** 从单个 SSE delta 中提取可展示正文；reasoning 不向编辑器输出 */
export const extractStreamDeltaText = (delta: unknown): { text: string; reasoningOnly: boolean } => {
  if (!delta || typeof delta !== 'object') return { text: '', reasoningOnly: false }
  const item = delta as Record<string, unknown>
  const text = extractTextContent(item.content)
  if (text) return { text, reasoningOnly: false }

  const reasoning = extractTextContent(item.reasoning_content)
  return { text: '', reasoningOnly: Boolean(reasoning) }
}

export const normalizeAiVisibleText = (text: string): string => String(text || '').trim()
