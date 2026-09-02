import { describe, expect, it } from 'vitest'
import {
  extractStreamDeltaText,
  normalizeChatCompletionResponse,
} from './ai-response-normalizer'

describe('ai-response-normalizer', () => {
  it('reads normal OpenAI message content', () => {
    expect(normalizeChatCompletionResponse({
      choices: [{ message: { content: '你好，世界' } }],
    })).toEqual({ text: '你好，世界', reasoningOnly: false })
  })

  it('reads content part arrays', () => {
    expect(normalizeChatCompletionResponse({
      choices: [{ message: { content: [{ type: 'text', text: '你好' }, { type: 'text', text: '世界' }] } }],
    })).toEqual({ text: '你好世界', reasoningOnly: false })
  })

  it('recognizes reasoning-only responses as empty user-visible output', () => {
    expect(normalizeChatCompletionResponse({
      choices: [{ message: { reasoning_content: '思考过程' } }],
    })).toEqual({ text: '', reasoningOnly: true })
  })

  it('supports direct choice.text compatibility responses', () => {
    expect(normalizeChatCompletionResponse({
      choices: [{ text: '兼容正文' }],
    })).toEqual({ text: '兼容正文', reasoningOnly: false })
  })

  it('extracts stream content and ignores reasoning_content', () => {
    expect(extractStreamDeltaText({ reasoning_content: '思考' })).toEqual({ text: '', reasoningOnly: true })
    expect(extractStreamDeltaText({ content: '正文' })).toEqual({ text: '正文', reasoningOnly: false })
  })
})
