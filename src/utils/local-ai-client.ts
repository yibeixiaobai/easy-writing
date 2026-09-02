import type {
  UserAiModelSavePayload,
  UserAiModelTestResult,
  UserAiRemoteModelListResult,
} from '@/types/user-ai-model'
import { createThinkStreamFilter, stripThinkBlocks } from '@/utils/ai-think-filter'
import { getLocalAiModelSecret, localAiModelCode, type LocalAiModel } from '@/storage/local-ai-models'
import { appendLocalAiRecord, estimateTokens } from '@/storage/local-ai-records'
import { isTauriRuntime } from '@/storage'
import { extractStreamDeltaText, normalizeChatCompletionResponse } from '@/utils/ai-response-normalizer'

const REQUEST_TIMEOUT_MS = 20_000
const COMPLETION_TIMEOUT_MS = 90_000
const STREAM_CONNECT_TIMEOUT_MS = 60_000
const STREAM_IDLE_TIMEOUT_MS = 90_000
const NON_STREAM_MIN_TOKENS = 2048

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

const resolveAiFetch = async (): Promise<FetchLike> => {
  if (isTauriRuntime()) {
    const { fetch: tauriFetch } = await import('@tauri-apps/plugin-http')
    return tauriFetch as unknown as FetchLike
  }
  return window.fetch.bind(window)
}

const isDashScope = (baseUrl: string) => String(baseUrl || '').includes('aliyuncs.com')
const isOpenAiOfficial = (baseUrl: string) => String(baseUrl || '').includes('api.openai.com')
const isOpenAiReasoningModel = (modelCode: string) => /^(o\d|gpt-5)/i.test(String(modelCode || '').trim())

export const buildChatBody = (params: {
  baseUrl: string
  modelCode: string
  messages: LocalChatMessageInput[]
  maxTokens?: number
  temperature?: number
  stream: boolean
}): Record<string, unknown> => {
  const body: Record<string, unknown> = { model: params.modelCode, messages: params.messages, stream: params.stream }
  const maxTokens = params.stream ? params.maxTokens : params.maxTokens ? Math.max(params.maxTokens, NON_STREAM_MIN_TOKENS) : undefined
  if (maxTokens) body[isOpenAiOfficial(params.baseUrl) ? 'max_completion_tokens' : 'max_tokens'] = maxTokens
  const dropTemperature = isOpenAiOfficial(params.baseUrl) && isOpenAiReasoningModel(params.modelCode)
  if (params.temperature !== undefined && !dropTemperature) body.temperature = params.temperature
  if (!params.stream && isDashScope(params.baseUrl)) body.enable_thinking = false
  return body
}

const buildAuthHeaders = (apiKey: string | undefined, withJson = true): Record<string, string> => {
  const headers: Record<string, string> = {}
  if (withJson) headers['Content-Type'] = 'application/json'
  const key = String(apiKey || '').trim()
  if (key) headers.Authorization = `Bearer ${key}`
  return headers
}

export const joinAiUrl = (baseUrl: string, path: string) => `${String(baseUrl || '').trim().replace(/\/+$/, '')}/${String(path).replace(/^\/+/, '')}`

const resolveRequestConfig = (payload: Partial<UserAiModelSavePayload>) => {
  let baseUrl = String(payload.baseUrl || '').trim()
  let apiKey = String(payload.apiKey || '').trim()
  let modelCode = String(payload.modelCode || '').trim()
  if (payload.id != null && (!apiKey || !baseUrl || !modelCode)) {
    const stored = getLocalAiModelSecret(localAiModelCode(payload.id))
    if (stored) {
      if (!apiKey) apiKey = stored.apiKey || ''
      if (!baseUrl) baseUrl = String(stored.baseUrl || '').trim()
      if (!modelCode) modelCode = String(stored.modelCode || '').trim()
    }
  }
  return { baseUrl, apiKey, modelCode }
}

const readableRequestError = (error: unknown): string => {
  if (error instanceof DOMException && error.name === 'AbortError') return `请求超时（${REQUEST_TIMEOUT_MS / 1000} 秒无响应）`
  if (error instanceof TypeError) return isTauriRuntime() ? '网络请求失败，请检查接口地址与网络' : '网络请求失败：可能是接口地址不对，或该供应商不允许网页端直连（浏览器跨域限制），桌面版不受此限制'
  return error instanceof Error ? error.message : '请求失败'
}

const readableHttpError = async (response: Response): Promise<string> => {
  let detail = ''
  try {
    const body = await response.json()
    detail = String(body?.error?.message || body?.message || '')
  } catch {}
  const byStatus: Record<number, string> = { 401: 'API Key 无效或未授权', 402: '账户余额不足，请到供应商后台充值', 403: '没有访问权限（检查 Key 的可用范围）', 404: '接口路径或模型不存在（检查 BaseURL 与模型名）', 429: '触发限流或额度不足' }
  const base = byStatus[response.status] || `请求失败（HTTP ${response.status}）`
  return detail ? `${base}：${detail.slice(0, 200)}` : base
}

const withTimeout = async <T>(run: (signal: AbortSignal) => Promise<T>): Promise<T> => {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try { return await run(controller.signal) } finally { window.clearTimeout(timer) }
}

/** 测试必须验证真正的 assistant 正文，而不是仅验证 HTTP 200。 */
export const testLocalAiModel = async (payload: Partial<UserAiModelSavePayload>): Promise<{ data: UserAiModelTestResult }> => {
  const { baseUrl, apiKey, modelCode } = resolveRequestConfig(payload)
  const url = joinAiUrl(baseUrl, 'chat/completions')
  const startedAt = Date.now()
  const result = (ok: boolean, message: string): { data: UserAiModelTestResult } => ({ data: { ok, message, latency: Date.now() - startedAt, url, testedAt: new Date().toISOString() } })
  if (!baseUrl) return result(false, '请先填写接口地址（BaseURL）')
  if (!modelCode) return result(false, '请先填写模型名称（modelCode）')
  try {
    const aiFetch = await resolveAiFetch()
    const response = await withTimeout(signal => aiFetch(url, { method: 'POST', signal, headers: buildAuthHeaders(apiKey), body: JSON.stringify(buildChatBody({ baseUrl, modelCode, messages: [{ role: 'user', content: '连通性测试，请回复"ok"' }], maxTokens: 16, stream: false })) }))
    if (!response.ok) return result(false, await readableHttpError(response))
    const body = await response.json()
    if (body?.error?.message) return result(false, `模型接口返回错误：${String(body.error.message).slice(0, 200)}`)
    const normalized = normalizeChatCompletionResponse(body)
    const content = stripThinkBlocks(normalized.text).trim()
    if (!content) return result(false, normalized.reasoningOnly ? '请求成功，但模型只返回了 reasoning/思考内容，没有可用正文' : '请求成功，但模型没有返回可用正文（请检查模型响应格式或模型配置）')
    return result(true, `连接成功，模型可用：${content.slice(0, 80)}`)
  } catch (error) { return result(false, readableRequestError(error)) }
}

export const NO_MODEL_MESSAGE = '还没有可用模型：请先到「模型管理」添加并启用一个文本模型'
export interface LocalAiSceneTag { scene?: string; sceneLabel?: string }
export interface LocalChatMessageInput { role: 'system' | 'user' | 'assistant'; content: string }
const messagesToText = (messages: LocalChatMessageInput[]) => messages.map(message => message.content).join('\n')

const recordAiCall = (entry: { recordType: 'text' | 'image'; tag: LocalAiSceneTag | undefined; model: LocalAiModel; status: 0 | 1; input: string; output: string; inputTokens?: number; outputTokens?: number; startedAt: number; errorMsg?: string }) => {
  void appendLocalAiRecord({ recordType: entry.recordType, scene: entry.tag?.scene || (entry.recordType === 'image' ? 'image_common' : 'text_common'), sceneLabel: entry.tag?.sceneLabel || (entry.recordType === 'image' ? '生图' : '文本生成'), modelCode: localAiModelCode(entry.model.id), modelName: entry.model.name || entry.model.modelCode, status: entry.status, input: entry.input, output: entry.output, inputTokens: entry.inputTokens ?? estimateTokens(entry.input), outputTokens: entry.outputTokens ?? estimateTokens(entry.output), duration: Date.now() - entry.startedAt, errorMsg: entry.errorMsg }).catch(error => console.warn('AI 调用记账失败', error))
}

export const requestLocalChatCompletion = async (options: { modelCode: string; messages: LocalChatMessageInput[]; maxTokens?: number; temperature?: number; signal?: AbortSignal } & LocalAiSceneTag): Promise<string> => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) throw new Error(NO_MODEL_MESSAGE)
  if (!model.baseUrl || !model.modelCode) throw new Error(`模型「${model.name}」配置不完整，请到模型管理检查`)
  const startedAt = Date.now()
  const recordInput = messagesToText(options.messages)
  const controller = new AbortController()
  let timedOut = false
  const timer = window.setTimeout(() => { timedOut = true; controller.abort() }, COMPLETION_TIMEOUT_MS)
  const onCallerAbort = () => controller.abort()
  if (options.signal) { if (options.signal.aborted) controller.abort(); else options.signal.addEventListener('abort', onCallerAbort, { once: true }) }
  try {
    const aiFetch = await resolveAiFetch()
    const response = await aiFetch(joinAiUrl(model.baseUrl, 'chat/completions'), { method: 'POST', signal: controller.signal, headers: buildAuthHeaders(model.apiKey), body: JSON.stringify(buildChatBody({ baseUrl: model.baseUrl, modelCode: model.modelCode, messages: options.messages, maxTokens: options.maxTokens || model.maxOutputTokens || undefined, temperature: options.temperature, stream: false })) })
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json()
    if (body?.error?.message) throw new Error(String(body.error.message))
    const normalized = normalizeChatCompletionResponse(body)
    const content = stripThinkBlocks(normalized.text).trim()
    if (!content) throw new Error(normalized.reasoningOnly ? '模型只返回了 reasoning/思考内容，没有返回可用正文' : '模型请求成功，但没有返回可用正文，请检查模型响应格式、模型配置或服务商兼容性')
    recordAiCall({ recordType: 'text', tag: options, model, status: 1, input: recordInput, output: content, inputTokens: Number(body?.usage?.prompt_tokens) || undefined, outputTokens: Number(body?.usage?.completion_tokens) || undefined, startedAt })
    return content
  } catch (error) {
    const readable = error instanceof DOMException && error.name === 'AbortError' ? (timedOut ? new Error(`生成超时（${COMPLETION_TIMEOUT_MS / 1000} 秒无结果），请重试`) : error) : new Error(readableRequestError(error))
    if (!(readable instanceof DOMException)) recordAiCall({ recordType: 'text', tag: options, model, status: 0, input: recordInput, output: '', outputTokens: 0, startedAt, errorMsg: readable.message })
    throw readable
  } finally { window.clearTimeout(timer); if (options.signal) options.signal.removeEventListener('abort', onCallerAbort) }
}

export interface LocalChatStreamCallbacks { onDelta: (text: string) => void; onDone: () => void; onError: (message: string) => void }

export const streamLocalChatCompletion = async (options: { modelCode: string; messages: LocalChatMessageInput[]; temperature?: number; signal?: AbortSignal } & LocalAiSceneTag, callbacks: LocalChatStreamCallbacks) => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) return callbacks.onError(NO_MODEL_MESSAGE)
  if (!model.baseUrl || !model.modelCode) return callbacks.onError(`模型「${model.name}」配置不完整，请到模型管理检查`)
  const startedAt = Date.now(), recordInput = messagesToText(options.messages)
  let collected = '', recorded = false
  const recordStream = (status: 0 | 1, errorMsg?: string) => { if (recorded) return; recorded = true; recordAiCall({ recordType: 'text', tag: options, model, status, input: recordInput, output: collected, startedAt, errorMsg }) }
  const controller = new AbortController()
  let timedOut = false, idleTimer: number | null = null
  const clearIdle = () => { if (idleTimer !== null) { window.clearTimeout(idleTimer); idleTimer = null } }
  const armIdle = (ms: number) => { clearIdle(); idleTimer = window.setTimeout(() => { timedOut = true; controller.abort() }, ms) }
  const onCallerAbort = () => controller.abort()
  if (options.signal) { if (options.signal.aborted) controller.abort(); else options.signal.addEventListener('abort', onCallerAbort, { once: true }) }
  try {
    const aiFetch = await resolveAiFetch(); armIdle(STREAM_CONNECT_TIMEOUT_MS)
    const response = await aiFetch(joinAiUrl(model.baseUrl, 'chat/completions'), { method: 'POST', signal: controller.signal, headers: buildAuthHeaders(model.apiKey), body: JSON.stringify(buildChatBody({ baseUrl: model.baseUrl, modelCode: model.modelCode, messages: options.messages, maxTokens: model.maxOutputTokens || undefined, temperature: options.temperature, stream: true })) })
    if (!response.ok) { const message = await readableHttpError(response); recordStream(0, message); callbacks.onError(message); return }
    if (!response.body) { const message = '当前环境不支持流式读取'; recordStream(0, message); callbacks.onError(message); return }
    const reader = response.body.getReader(), decoder = new TextDecoder('utf-8'), thinkFilter = createThinkStreamFilter()
    let buffer = '', finished = false, sawData = false, sawReasoning = false
    while (true) {
      const { done, value } = await reader.read(); if (done) break
      armIdle(STREAM_IDLE_TIMEOUT_MS); buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split(/\r?\n/); buffer = lines.pop() || ''
      for (const rawLine of lines) {
        const line = rawLine.trim(); if (!line.startsWith('data:')) continue
        sawData = true; const payload = line.slice(5).trim()
        if (payload === '[DONE]') { finished = true; break }
        try {
          const chunk = JSON.parse(payload)
          if (chunk?.error?.message) { const message = String(chunk.error.message); recordStream(0, message); callbacks.onError(message); return }
          const extracted = extractStreamDeltaText(chunk?.choices?.[0]?.delta)
          sawReasoning = sawReasoning || extracted.reasoningOnly
          if (extracted.text) { const visible = thinkFilter.push(extracted.text); if (visible) { collected += visible; callbacks.onDelta(visible) } }
        } catch {}
      }
      if (finished) break
    }
    const trailing = buffer.trim()
    if (trailing.startsWith('data:')) {
      const payload = trailing.slice(5).trim()
      if (payload !== '[DONE]') { try { const chunk = JSON.parse(payload); const extracted = extractStreamDeltaText(chunk?.choices?.[0]?.delta); sawReasoning = sawReasoning || extracted.reasoningOnly; if (extracted.text) { const visible = thinkFilter.push(extracted.text); if (visible) { collected += visible; callbacks.onDelta(visible) } } } catch {} }
    }
    const tail = thinkFilter.finish(); if (tail) { collected += tail; callbacks.onDelta(tail) }
    collected = stripThinkBlocks(collected).trim()
    if (!collected) { const message = sawReasoning ? '模型只返回了 reasoning/思考内容，没有返回可用正文' : sawData ? '模型请求成功，但流式响应没有返回可用正文' : '模型请求成功，但没有收到有效的 SSE 数据'; recordStream(0, message); callbacks.onError(message); return }
    recordStream(1); callbacks.onDone()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (timedOut) { recordStream(0, 'AI 响应超时'); callbacks.onError('AI 响应超时，请重试') }
      else { recordStream(collected ? 1 : 0, collected ? undefined : '用户中止且未产生可用内容'); callbacks.onDone() }
      return
    }
    const message = readableRequestError(error); recordStream(0, message); callbacks.onError(message)
  } finally { clearIdle(); if (options.signal) options.signal.removeEventListener('abort', onCallerAbort) }
}

export const listLocalAiRemoteModels = async (payload: Partial<UserAiModelSavePayload>): Promise<{ data: UserAiRemoteModelListResult }> => {
  const { baseUrl, apiKey } = resolveRequestConfig(payload), url = joinAiUrl(baseUrl, 'models'), startedAt = Date.now()
  if (!baseUrl) throw new Error('请先填写接口地址（BaseURL）')
  try {
    const aiFetch = await resolveAiFetch(), response = await withTimeout(signal => aiFetch(url, { method: 'GET', signal, headers: buildAuthHeaders(apiKey, false) }))
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json(), rawList = Array.isArray(body?.data) ? body.data : Array.isArray(body) ? body : []
    const models = rawList.map((item: { id?: unknown }) => String(item?.id || '').trim()).filter(Boolean).sort((a: string, b: string) => a.localeCompare(b))
    return { data: { models, total: models.length, url, latency: Date.now() - startedAt, testedAt: new Date().toISOString() } }
  } catch (error) { const message = readableRequestError(error); throw new Error(message === '请求失败' ? '拉取模型清单失败' : message) }
}

const IMAGE_TIMEOUT_MS = 300_000
export interface LocalAiImageResult { blob?: Blob; remoteUrl?: string }
const base64ToBlob = (b64: string, type = 'image/png') => { const binary = atob(b64), bytes = new Uint8Array(binary.length); for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index); return new Blob([bytes], { type }) }

export const generateLocalAiImageRequest = async (options: { modelCode: string; prompt: string; size?: string; quality?: string; signal?: AbortSignal } & LocalAiSceneTag): Promise<LocalAiImageResult> => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) throw new Error(NO_MODEL_MESSAGE)
  if (!model.baseUrl || !model.modelCode) throw new Error(`模型「${model.name}」配置不完整，请到模型管理检查`)
  const startedAt = Date.now()
  const recordImage = (status: 0 | 1, errorMsg?: string) => recordAiCall({ recordType: 'image', tag: options, model, status, input: options.prompt, output: '', inputTokens: 0, outputTokens: 0, startedAt, errorMsg })
  const controller = new AbortController(); let timedOut = false
  const timer = window.setTimeout(() => { timedOut = true; controller.abort() }, IMAGE_TIMEOUT_MS)
  const onCallerAbort = () => controller.abort()
  if (options.signal) { if (options.signal.aborted) controller.abort(); else options.signal.addEventListener('abort', onCallerAbort, { once: true }) }
  try {
    const aiFetch = await resolveAiFetch(), response = await aiFetch(joinAiUrl(model.baseUrl, 'images/generations'), { method: 'POST', signal: controller.signal, headers: buildAuthHeaders(model.apiKey), body: JSON.stringify({ model: model.modelCode, prompt: options.prompt, n: 1, response_format: 'b64_json', ...(options.size ? { size: options.size } : {}), ...(options.quality ? { quality: options.quality } : {}) }) })
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json(); if (body?.error?.message) throw new Error(String(body.error.message))
    const item = body?.data?.[0] || {}, b64 = String(item.b64_json || '')
    if (b64) { recordImage(1); return { blob: base64ToBlob(b64) } }
    const url = String(item.url || ''); if (!url) throw new Error('生图接口没有返回图片数据')
    try { const imageResponse = await aiFetch(url, { method: 'GET', signal: controller.signal }); if (!imageResponse.ok) throw new Error(`HTTP ${imageResponse.status}`); recordImage(1); return { blob: await imageResponse.blob() } }
    catch { recordImage(1); return { remoteUrl: url } }
  } catch (error) {
    const readable = error instanceof DOMException && error.name === 'AbortError' ? (timedOut ? new Error(`生图超时（${IMAGE_TIMEOUT_MS / 1000} 秒无结果），请重试`) : error) : new Error(readableRequestError(error))
    if (!(readable instanceof DOMException)) recordImage(0, readable.message)
    throw readable
  } finally { window.clearTimeout(timer); if (options.signal) options.signal.removeEventListener('abort', onCallerAbort) }
}
