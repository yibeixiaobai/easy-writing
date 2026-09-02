/** Normalize common OpenAI-compatible chat responses. */
export interface NormalizedTextResult {
  text: string
  reasoningOnly: boolean
}

export const extractTextContent = (value: unknown): string => {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value.map(part => {
    if (typeof part === 'string') return part
    if (!part || typeof part !== 'object') return ''
    const item = part as Record<string, unknown>
    return typeof item.text === 'string' ? item.text : typeof item.content === 'string' ? item.content : ''
  }).join('')
}

export const normalizeChatCompletionResponse = (body: unknown): NormalizedTextResult => {
  if (!body || typeof body !== 'object') return { text: '', reasoningOnly: false }
  const root = body as Record<string, unknown>
  const choices = Array.isArray(root.choices) ? root.choices : []
  const first = choices[0]
  if (!first || typeof first !== 'object') return { text: '', reasoningOnly: false }
  const choice = first as Record<string, unknown>
  const message = choice.message && typeof choice.message === 'object' ? choice.message as Record<string, unknown> : null
  const content = extractTextContent(message?.content)
  if (content.trim()) return { text: content, reasoningOnly: false }
  const choiceText = typeof choice.text === 'string' ? choice.text : ''
  if (choiceText.trim()) return { text: choiceText, reasoningOnly: false }
  const reasoning = extractTextContent(message?.reasoning_content)
  return { text: '', reasoningOnly: Boolean(reasoning.trim()) }
}

export const extractStreamDeltaText = (delta: unknown): { text: string; reasoningOnly: boolean } => {
  if (!delta || typeof delta !== 'object') return { text: '', reasoningOnly: false }
  const item = delta as Record<string, unknown>
  const text = extractTextContent(item.content)
  if (text) return { text, reasoningOnly: false }
  const reasoning = extractTextContent(item.reasoning_content)
  return { text: '', reasoningOnly: Boolean(reasoning.trim()) }
}
