import type { AuthTokens } from './types'

type JsonRecord = Record<string, unknown>
type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export interface StoredSession {
  accessToken: string
  refreshToken: string
}

interface RequestOptions extends RequestInit {
  skipAuth?: boolean
  skipRefresh?: boolean
}

const SESSION_KEY = 'criminal-defense-mvp-session'

function getStorage(): Storage | null {
  if (typeof window === 'undefined') return null
  const storage = window.localStorage
  return typeof storage?.getItem === 'function' ? storage : null
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function unwrapPayload<T>(payload: unknown): T {
  if (
    payload &&
    typeof payload === 'object' &&
    'data' in payload &&
    ('code' in payload || 'message' in payload || Object.keys(payload).length === 1)
  ) {
    return (payload as { data: T }).data
  }
  return payload as T
}

function readError(payload: unknown, status: number): ApiError {
  if (payload && typeof payload === 'object' && 'error' in payload) {
    const error = (payload as { error?: { code?: string; message?: string } }).error
    return new ApiError(status, error?.code ?? 'HTTP_ERROR', error?.message ?? '请求失败')
  }
  return new ApiError(status, 'HTTP_ERROR', `请求失败（HTTP ${status}）`)
}

export function getUserErrorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return error instanceof TypeError ? '无法连接后端，请检查服务是否启动或网络是否可用。' : '操作未完成，请稍后重试。'
  }
  if (error.status === 401) return error.message.includes('用户名或密码') ? error.message : '登录状态已失效，请重新登录。'
  if (error.status === 403) return error.message || '当前账号无权执行此操作。'
  if (error.status === 422) return '填写内容未通过校验，请检查必填项与格式。'
  if (error.status === 429) return '操作过于频繁，请稍后再试。'
  if (error.status === 502 || error.code === 'LLM_SERVICE_ERROR') return '模型服务暂时不可用，本次内容尚未处理。请稍后重试。'
  if (error.status === 504 || error.code === 'LLM_TIMEOUT') return '模型服务响应超时，本次内容尚未处理。请稍后重试。'
  return error.message || '操作未完成，请稍后重试。'
}

export class ApiClient {
  private session: StoredSession | null = null
  private onSessionExpired?: () => void

  constructor(
    private readonly baseUrl = '/api/v1',
    private readonly fetcher: Fetcher = fetch.bind(globalThis),
  ) {
    const storage = getStorage()
    try {
      const raw = storage?.getItem(SESSION_KEY)
      this.session = raw ? (JSON.parse(raw) as StoredSession) : null
    } catch {
      storage?.removeItem(SESSION_KEY)
    }
  }

  setSession(session: StoredSession): void {
    this.session = session
    getStorage()?.setItem(SESSION_KEY, JSON.stringify(session))
  }

  getSession(): StoredSession | null {
    return this.session
  }

  clearSession(): void {
    this.session = null
    getStorage()?.removeItem(SESSION_KEY)
  }

  setSessionExpiredHandler(handler: () => void): void {
    this.onSessionExpired = handler
  }

  async get<T>(path: string): Promise<T> {
    return this.request<T>(path)
  }

  async post<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>(path, { ...options, method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
  }

  async put<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, { method: 'PUT', body: JSON.stringify(body) })
  }

  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const headers = new Headers(options.headers)
    if (options.body !== undefined) headers.set('Content-Type', 'application/json')
    if (!options.skipAuth && this.session?.accessToken) {
      headers.set('Authorization', `Bearer ${this.session.accessToken}`)
    }

    let response: Response
    try {
      response = await this.fetcher(`${this.baseUrl}${path}`, { ...options, headers })
    } catch (error) {
      throw error instanceof TypeError ? error : new TypeError('Network request failed')
    }

    if (response.status === 401 && !options.skipRefresh && !options.skipAuth && this.session?.refreshToken) {
      const refreshed = await this.refreshTokens()
      if (refreshed) return this.request<T>(path, { ...options, skipRefresh: true })
      this.clearSession()
      this.onSessionExpired?.()
    }

    const payload = await response.json().catch(() => ({} as JsonRecord))
    if (!response.ok) throw readError(payload, response.status)
    return unwrapPayload<T>(payload)
  }

  private async refreshTokens(): Promise<boolean> {
    const refreshToken = this.session?.refreshToken
    if (!refreshToken) return false
    try {
      const response = await this.fetcher(`${this.baseUrl}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
      if (!response.ok) return false
      const tokens = unwrapPayload<AuthTokens>(await response.json())
      this.setSession({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token })
      return true
    } catch {
      return false
    }
  }
}

export const api = new ApiClient()
