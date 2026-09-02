import type {
  UserAiModelSavePayload,
  UserAiModelTestResult,
  UserAiRemoteModelListResult,
} from '@/types/user-ai-model'
import { createThinkStreamFilter, stripThinkBlocks } from '@/utils/ai-think-filter'
import { getLocalAiModelSecret, localAiModelCode, type LocalAiModel } from '@/storage/local-ai-models'
import { appendLocalAiRecord, estimateTokens } from '@/storage/local-ai-records'
import { isTauriRuntime } from '@/storage'

/**
 * BYOK 直连请求层（OpenAI 兼容协议）：密钥只在本机内存/存储流转，请求直发供应商。
 *
 * - 桌面端走 @tauri-apps/plugin-http 的 fetch（不受浏览器跨域限制）。
 * - 网页端走浏览器 fetch：部分供应商允许浏览器直连，不允许的会被浏览器拦下，
 *   报错文案会提示改用桌面版。
 */

const REQUEST_TIMEOUT_MS = 20000

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

const resolveAiFetch = async (): Promise<FetchLike> => {
  if (isTauriRuntime()) {
    const { fetch: tauriFetch } = await import('@tauri-apps/plugin-http')
    // 插件会给每个请求自动补 Origin 头（Windows 是 http://tauri.localhost），
    // Ollama 等本地服务会按来源拒绝（403）。传空串 = 让插件摘掉这个头，
    // 桌面端请求从此与 curl 一样不带来源。需 Cargo 给插件开 unsafe-headers 特性。
    return (input, init) =>
      tauriFetch(input, {
        ...init,
        headers: { Origin: '', ...(init?.headers as Record<string, string> | undefined) },
      })
  }
  return window.fetch.bind(window)
}

/** 阿里 dashscope 接口：思考型千问模型走非流式必须显式关思考，否则服务端直接 400
 *（官方要求 enable_thinking=false 或改用流式；按 baseUrl 判断，自定义填法也能盖住） */
const isDashScope = (baseUrl: string) => String(baseUrl || '').includes('aliyuncs.com')

// OpenAI 官方接口的两个换代差异（其余兼容渠道仍认老字段）：
// token 上限字段改名 max_completion_tokens；推理系（o*/gpt-5*）只认默认温度
const isOpenAiOfficial = (baseUrl: string) => String(baseUrl || '').includes('api.openai.com')
const isOpenAiReasoningModel = (modelCode: string) => /^(o\d|gpt-5)/i.test(String(modelCode || '').trim())

// 非流式最少给足 2048 token 上限：思考型模型（Gemini flash、DeepSeek flash 等）把
// 思考计入输出上限，预算太小会被思考吃光，拿回空正文或半截 JSON（实测 1024 仍不够）；
// 上限只是护栏，普通模型不会因此多产出
const NON_STREAM_MIN_TOKENS = 2048

/** 按供应商差异拼 chat/completions 请求体：三家怪癖集中在这一处 */
export const buildChatBody = (params: {
  baseUrl: string
  modelCode: string
  messages: LocalChatMessageInput[]
  maxTokens?: number
  temperature?: number
  stream: boolean
}): Record<string, unknown> => {
  const body: Record<string, unknown> = {
    model: params.modelCode,
    messages: params.messages,
    stream: params.stream,
  }
  const maxTokens = params.stream
    ? params.maxTokens
    : params.maxTokens
      ? Math.max(params.maxTokens, NON_STREAM_MIN_TOKENS)
      : undefined
  if (maxTokens) {
    body[isOpenAiOfficial(params.baseUrl) ? 'max_completion_tokens' : 'max_tokens'] = maxTokens
  }
  const dropTemperature = isOpenAiOfficial(params.baseUrl) && isOpenAiReasoningModel(params.modelCode)
  if (params.temperature !== undefined && !dropTemperature) {
    body.temperature = params.temperature
  }
  // 阿里 dashscope：思考型千问走非流式必须显式关思考，否则服务端直接 400
  if (!params.stream && isDashScope(params.baseUrl)) {
    body.enable_thinking = false
  }
  return body
}

/** 本地部署（Ollama/LM Studio 等）通常没有 API Key：留空则不带鉴权头 */
const buildAuthHeaders = (apiKey: string | undefined, withJson = true): Record<string, string> => {
  const headers: Record<string, string> = {}
  if (withJson) headers['Content-Type'] = 'application/json'
  const key = String(apiKey || '').trim()
  if (key) headers.Authorization = `Bearer ${key}`
  return headers
}

/** baseUrl 与端点拼接：只负责去重斜杠，版本段（/v1 等）以用户填写为准 */
export const joinAiUrl = (baseUrl: string, path: string) =>
  `${String(baseUrl || '').trim().replace(/\/+$/, '')}/${String(path).replace(/^\/+/, '')}`

/** 把测试/拉取用的载荷补齐配置：字段留空且带 id 时回查本地存储。
 *  覆盖两种调用：编辑表单（密钥留空=不修改）与列表行内测试（只传 id）。 */
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
  if (error instanceof DOMException && error.name === 'AbortError') {
    return `请求超时（${REQUEST_TIMEOUT_MS / 1000} 秒无响应）`
  }
  if (error instanceof TypeError) {
    return isTauriRuntime()
      ? '网络请求失败，请检查接口地址与网络'
      : '网络请求失败：可能是接口地址不对，或该供应商不允许网页端直连（浏览器跨域限制），桌面版不受此限制'
  }
  return error instanceof Error ? error.message : '请求失败'
}

const readableHttpError = async (response: Response): Promise<string> => {
  let detail = ''
  try {
    const body = await response.json()
    detail = String(body?.error?.message || body?.message || '')
  } catch {
    // 响应体不是 JSON 时只按状态码给文案
  }
  const byStatus: Record<number, string> = {
    401: 'API Key 无效或未授权',
    402: '账户余额不足，请到供应商后台充值',
    403: '没有访问权限：云端服务请检查 Key 的可用范围，本地服务请检查它的来源或 IP 访问限制',
    404: '接口路径或模型不存在（检查 BaseURL 与模型名）',
    429: '触发限流或额度不足',
  }
  const base = byStatus[response.status] || `请求失败（HTTP ${response.status}）`
  return detail ? `${base}：${detail.slice(0, 200)}` : base
}

const withTimeout = async <T>(run: (signal: AbortSignal) => Promise<T>): Promise<T> => {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    return await run(controller.signal)
  } finally {
    window.clearTimeout(timer)
  }
}

/** 连通测试：发一条最小 chat 请求验证 BaseURL/Key/模型名三件事 */
export const testLocalAiModel = async (
  payload: Partial<UserAiModelSavePayload>
): Promise<{ data: UserAiModelTestResult }> => {
  const { baseUrl, apiKey, modelCode } = resolveRequestConfig(payload)
  const url = joinAiUrl(baseUrl, 'chat/completions')
  const startedAt = Date.now()
  const result = (ok: boolean, message: string): { data: UserAiModelTestResult } => ({
    data: { ok, message, latency: Date.now() - startedAt, url, testedAt: new Date().toISOString() },
  })
  if (!baseUrl) return result(false, '请先填写接口地址（BaseURL）')
  if (!modelCode) return result(false, '请先填写模型名称（modelCode）')

  try {
    const aiFetch = await resolveAiFetch()
    const response = await withTimeout(signal =>
      aiFetch(url, {
        method: 'POST',
        signal,
        headers: buildAuthHeaders(apiKey),
        body: JSON.stringify(
          buildChatBody({
            baseUrl,
            modelCode,
            messages: [{ role: 'user', content: '连通性测试，请回复"ok"' }],
            maxTokens: 16,
            stream: false,
          })
        ),
      })
    )
    if (!response.ok) return result(false, await readableHttpError(response))
    return result(true, '连接成功，模型可用')
  } catch (error) {
    return result(false, readableRequestError(error))
  }
}

export const NO_MODEL_MESSAGE = '还没有可用模型：请先到「模型管理」添加并启用一个文本模型'

/** 调用方标注的场景（进「AI 调用记录」账本）；不传按类型给通用标签 */
export interface LocalAiSceneTag {
  scene?: string
  sceneLabel?: string
}

const messagesToText = (messages: LocalChatMessageInput[]) =>
  messages.map(message => message.content).join('\n')

/** 落账永不影响调用本身：任何记账异常只进控制台 */
const recordAiCall = (entry: {
  recordType: 'text' | 'image'
  tag: LocalAiSceneTag | undefined
  model: LocalAiModel
  status: 0 | 1
  input: string
  output: string
  inputTokens?: number
  outputTokens?: number
  startedAt: number
  errorMsg?: string
}) => {
  void appendLocalAiRecord({
    recordType: entry.recordType,
    scene: entry.tag?.scene || (entry.recordType === 'image' ? 'image_common' : 'text_common'),
    sceneLabel: entry.tag?.sceneLabel || (entry.recordType === 'image' ? '生图' : '文本生成'),
    modelCode: localAiModelCode(entry.model.id),
    modelName: entry.model.name || entry.model.modelCode,
    status: entry.status,
    input: entry.input,
    output: entry.output,
    inputTokens: entry.inputTokens ?? estimateTokens(entry.input),
    outputTokens: entry.outputTokens ?? estimateTokens(entry.output),
    duration: Date.now() - entry.startedAt,
    errorMsg: entry.errorMsg,
  }).catch(error => console.warn('AI 调用记账失败', error))
}

// 非流式生成给足时间：润色/扩写可能一次产出几百字
const COMPLETION_TIMEOUT_MS = 90_000

/**
 * 非流式补全：一次性返回全文（划词润色/打字补全这类"拿到结果再落格"的场景）。
 * 失败抛出带可读文案的 Error；外部 signal 中止原样抛 AbortError 由调用方静默。
 */
export const requestLocalChatCompletion = async (options: {
  modelCode: string
  messages: LocalChatMessageInput[]
  maxTokens?: number
  /** 采样温度（0-2）：来自提示词库逐场景配置；未传用模型服务默认 */
  temperature?: number
  signal?: AbortSignal
} & LocalAiSceneTag): Promise<string> => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) throw new Error(NO_MODEL_MESSAGE)
  if (!model.baseUrl || !model.modelCode) {
    throw new Error(`模型「${model.name}」配置不完整，请到模型管理检查`)
  }
  const startedAt = Date.now()
  const recordInput = messagesToText(options.messages)

  const controller = new AbortController()
  let timedOut = false
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, COMPLETION_TIMEOUT_MS)
  const onCallerAbort = () => controller.abort()
  if (options.signal) {
    if (options.signal.aborted) controller.abort()
    else options.signal.addEventListener('abort', onCallerAbort, { once: true })
  }

  try {
    const aiFetch = await resolveAiFetch()
    const response = await aiFetch(joinAiUrl(model.baseUrl, 'chat/completions'), {
      method: 'POST',
      signal: controller.signal,
      headers: buildAuthHeaders(model.apiKey),
      body: JSON.stringify(
        buildChatBody({
          baseUrl: model.baseUrl,
          modelCode: model.modelCode,
          messages: options.messages,
          maxTokens: options.maxTokens || model.maxOutputTokens || undefined,
          temperature: options.temperature,
          stream: false,
        })
      ),
    })
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json()
    if (body?.error?.message) throw new Error(String(body.error.message))
    // 剥掉部分渠道内联进 content 的 <think> 思考段，只留真正文
    const content = stripThinkBlocks(String(body?.choices?.[0]?.message?.content || '')).trim()
    recordAiCall({
      recordType: 'text',
      tag: options,
      model,
      status: 1,
      input: recordInput,
      output: content,
      inputTokens: Number(body?.usage?.prompt_tokens) || undefined,
      outputTokens: Number(body?.usage?.completion_tokens) || undefined,
      startedAt,
    })
    return content
  } catch (error) {
    const readable =
      error instanceof DOMException && error.name === 'AbortError'
        ? timedOut
          ? new Error(`生成超时（${COMPLETION_TIMEOUT_MS / 1000} 秒无结果），请重试`)
          : error
        : new Error(readableRequestError(error))
    // 用户主动中止不算失败，不落账；其余失败如实记一笔
    if (!(readable instanceof DOMException)) {
      recordAiCall({
        recordType: 'text',
        tag: options,
        model,
        status: 0,
        input: recordInput,
        output: '',
        outputTokens: 0,
        startedAt,
        errorMsg: readable.message,
      })
    }
    throw readable
  } finally {
    window.clearTimeout(timer)
    if (options.signal) options.signal.removeEventListener('abort', onCallerAbort)
  }
}

// ---------------------------------------------------------------------------
// 流式对话（OpenAI 兼容 SSE：data: {choices:[{delta:{content}}]} … data: [DONE]）
// ---------------------------------------------------------------------------

// 首字节等待与分片间空闲上限：任一超时主动中止，避免连接挂起时界面永远"生成中"
const STREAM_CONNECT_TIMEOUT_MS = 60_000
const STREAM_IDLE_TIMEOUT_MS = 90_000

export interface LocalChatMessageInput {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface LocalChatStreamCallbacks {
  onDelta: (text: string) => void
  onDone: () => void
  onError: (message: string) => void
}

/**
 * 流式对话：按本地模型 code 取配置直连供应商。
 * 用户主动中止（signal）按正常收尾（onDone），超时中止报错——与旧流式层同语义。
 */
export const streamLocalChatCompletion = async (
  options: {
    modelCode: string
    messages: LocalChatMessageInput[]
    /** 采样温度（0-2）：来自提示词库逐场景配置；未传用模型服务默认 */
    temperature?: number
    signal?: AbortSignal
  } & LocalAiSceneTag,
  callbacks: LocalChatStreamCallbacks
) => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) {
    callbacks.onError(NO_MODEL_MESSAGE)
    return
  }
  if (!model.baseUrl || !model.modelCode) {
    callbacks.onError(`模型「${model.name}」配置不完整，请到模型管理检查`)
    return
  }
  const startedAt = Date.now()
  const recordInput = messagesToText(options.messages)
  let collected = ''
  let recorded = false
  const recordStream = (status: 0 | 1, errorMsg?: string) => {
    if (recorded) return
    recorded = true
    recordAiCall({
      recordType: 'text',
      tag: options,
      model,
      status,
      input: recordInput,
      output: collected,
      startedAt,
      errorMsg,
    })
  }

  const controller = new AbortController()
  let timedOut = false
  let idleTimer: number | null = null
  const clearIdle = () => {
    if (idleTimer) {
      window.clearTimeout(idleTimer)
      idleTimer = null
    }
  }
  const armIdle = (ms: number) => {
    clearIdle()
    idleTimer = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, ms)
  }
  const onCallerAbort = () => controller.abort()
  if (options.signal) {
    if (options.signal.aborted) controller.abort()
    else options.signal.addEventListener('abort', onCallerAbort, { once: true })
  }

  try {
    const aiFetch = await resolveAiFetch()
    armIdle(STREAM_CONNECT_TIMEOUT_MS)
    const response = await aiFetch(joinAiUrl(model.baseUrl, 'chat/completions'), {
      method: 'POST',
      signal: controller.signal,
      headers: buildAuthHeaders(model.apiKey),
      body: JSON.stringify(
        buildChatBody({
          baseUrl: model.baseUrl,
          modelCode: model.modelCode,
          messages: options.messages,
          maxTokens: model.maxOutputTokens || undefined,
          temperature: options.temperature,
          stream: true,
        })
      ),
    })
    if (!response.ok) {
      const message = await readableHttpError(response)
      recordStream(0, message)
      callbacks.onError(message)
      return
    }
    if (!response.body) {
      callbacks.onError('当前环境不支持流式读取')
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    // 渠道可能把 <think> 思考段内联进增量正文，跨分片过滤后再吐给调用方
    const thinkFilter = createThinkStreamFilter()
    let buffer = ''
    let finished = false
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      armIdle(STREAM_IDLE_TIMEOUT_MS)
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const rawLine of lines) {
        const line = rawLine.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') {
          finished = true
          break
        }
        try {
          const chunk = JSON.parse(payload)
          if (chunk?.error?.message) {
            const message = String(chunk.error.message)
            recordStream(0, message)
            callbacks.onError(message)
            return
          }
          const delta = chunk?.choices?.[0]?.delta?.content
          if (typeof delta === 'string' && delta) {
            const visible = thinkFilter.push(delta)
            if (visible) {
              collected += visible
              callbacks.onDelta(visible)
            }
          }
        } catch {
          // 跨分片被截断的 JSON 行极少见（按 \n 切已规避大半），忽略无法解析的行
        }
      }
      if (finished) break
    }
    const tail = thinkFilter.finish()
    if (tail) {
      collected += tail
      callbacks.onDelta(tail)
    }
    recordStream(1)
    callbacks.onDone()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      if (timedOut) {
        recordStream(0, 'AI 响应超时')
        callbacks.onError('AI 响应超时，请重试')
      } else {
        // 用户主动中止按正常收尾：已产出的部分如实入账
        recordStream(1)
        callbacks.onDone()
      }
      return
    }
    const message = readableRequestError(error)
    recordStream(0, message)
    callbacks.onError(message)
  } finally {
    clearIdle()
    if (options.signal) options.signal.removeEventListener('abort', onCallerAbort)
  }
}

/** 拉取供应商可用模型清单（GET {base}/models，OpenAI 兼容形状） */
export const listLocalAiRemoteModels = async (
  payload: Partial<UserAiModelSavePayload>
): Promise<{ data: UserAiRemoteModelListResult }> => {
  const { baseUrl, apiKey } = resolveRequestConfig(payload)
  const url = joinAiUrl(baseUrl, 'models')
  const startedAt = Date.now()
  if (!baseUrl) throw new Error('请先填写接口地址（BaseURL）')

  try {
    const aiFetch = await resolveAiFetch()
    const response = await withTimeout(signal =>
      aiFetch(url, {
        method: 'GET',
        signal,
        headers: buildAuthHeaders(apiKey, false),
      })
    )
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json()
    const rawList = Array.isArray(body?.data) ? body.data : Array.isArray(body) ? body : []
    const models = rawList
      .map((item: { id?: unknown }) => String(item?.id || '').trim())
      .filter(Boolean)
      .sort((a: string, b: string) => a.localeCompare(b))
    return {
      data: {
        models,
        total: models.length,
        url,
        latency: Date.now() - startedAt,
        testedAt: new Date().toISOString(),
      },
    }
  } catch (error) {
    const message = readableRequestError(error)
    throw new Error(message === '请求失败' ? '拉取模型清单失败' : message)
  }
}

// ---------------------------------------------------------------------------
// 生图（OpenAI 兼容 images/generations）
// ---------------------------------------------------------------------------

// 生图是长任务：gpt-image 常规 70~120s，给足 5 分钟；超时文案单独给
const IMAGE_TIMEOUT_MS = 300_000

export interface LocalAiImageResult {
  /** 优先：图片二进制（b64 响应或 url 已成功回捞） */
  blob?: Blob
  /** 兜底：仅拿到远程地址且网页端跨域捞不回（地址可能过期，调用方如实入库） */
  remoteUrl?: string
}

const base64ToBlob = (b64: string, type = 'image/png') => {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  return new Blob([bytes], { type })
}

/**
 * 按本地模型 code 直连供应商生图。请求 b64_json，供应商忽略该参数只回 url 时
 * 尝试把图捞回本地（桌面端不受跨域限制）；捞不回就退回 remoteUrl。
 */
export const generateLocalAiImageRequest = async (options: {
  modelCode: string
  prompt: string
  size?: string
  quality?: string
  signal?: AbortSignal
} & LocalAiSceneTag): Promise<LocalAiImageResult> => {
  const model = getLocalAiModelSecret(options.modelCode)
  if (!model) throw new Error(NO_MODEL_MESSAGE)
  if (!model.baseUrl || !model.modelCode) {
    throw new Error(`模型「${model.name}」配置不完整，请到模型管理检查`)
  }
  const startedAt = Date.now()
  const recordImage = (status: 0 | 1, errorMsg?: string) =>
    recordAiCall({
      recordType: 'image',
      tag: options,
      model,
      status,
      input: options.prompt,
      output: '',
      inputTokens: 0,
      outputTokens: 0,
      startedAt,
      errorMsg,
    })

  const controller = new AbortController()
  let timedOut = false
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, IMAGE_TIMEOUT_MS)
  const onCallerAbort = () => controller.abort()
  if (options.signal) {
    if (options.signal.aborted) controller.abort()
    else options.signal.addEventListener('abort', onCallerAbort, { once: true })
  }

  try {
    const aiFetch = await resolveAiFetch()
    const response = await aiFetch(joinAiUrl(model.baseUrl, 'images/generations'), {
      method: 'POST',
      signal: controller.signal,
      headers: buildAuthHeaders(model.apiKey),
      body: JSON.stringify({
        model: model.modelCode,
        prompt: options.prompt,
        n: 1,
        response_format: 'b64_json',
        ...(options.size ? { size: options.size } : {}),
        ...(options.quality ? { quality: options.quality } : {}),
      }),
    })
    if (!response.ok) throw new Error(await readableHttpError(response))
    const body = await response.json()
    if (body?.error?.message) throw new Error(String(body.error.message))
    const item = body?.data?.[0] || {}
    const b64 = String(item.b64_json || '')
    if (b64) {
      recordImage(1)
      return { blob: base64ToBlob(b64) }
    }
    const url = String(item.url || '')
    if (!url) throw new Error('生图接口没有返回图片数据')
    recordImage(1)
    try {
      const imageResponse = await aiFetch(url, { method: 'GET', signal: controller.signal })
      if (!imageResponse.ok) throw new Error(`HTTP ${imageResponse.status}`)
      return { blob: await imageResponse.blob() }
    } catch {
      // 网页端常见：图床跨域取不回二进制——退回远程地址，调用方如实标注可能过期
      return { remoteUrl: url }
    }
  } catch (error) {
    const readable =
      error instanceof DOMException && error.name === 'AbortError'
        ? timedOut
          ? new Error(`生图超时（${IMAGE_TIMEOUT_MS / 1000} 秒无结果），请重试`)
          : error
        : new Error(readableRequestError(error))
    if (!(readable instanceof DOMException)) recordImage(0, readable.message)
    throw readable
  } finally {
    window.clearTimeout(timer)
    if (options.signal) options.signal.removeEventListener('abort', onCallerAbort)
  }
}
