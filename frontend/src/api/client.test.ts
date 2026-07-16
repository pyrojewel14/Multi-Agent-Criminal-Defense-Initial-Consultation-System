import { ApiClient, ApiError, getUserErrorMessage, unwrapPayload } from './client'

describe('API contract helpers', () => {
  it('unwraps success envelopes and leaves bare workflow responses intact', () => {
    expect(unwrapPayload({ code: 200, message: 'success', data: { id: 'u-1' } })).toEqual({ id: 'u-1' })
    expect(unwrapPayload({ session_id: 'wf-1' })).toEqual({ session_id: 'wf-1' })
  })

  it('maps validation and LLM failures to actionable Chinese feedback', () => {
    expect(getUserErrorMessage(new ApiError(422, 'VALIDATION_ERROR', '请求参数校验失败'))).toContain('填写内容')
    expect(getUserErrorMessage(new ApiError(502, 'LLM_SERVICE_ERROR', '模型不可用'))).toContain('模型服务')
  })

  it('preserves credential feedback for login 401 responses', () => {
    expect(getUserErrorMessage(new ApiError(401, 'UNAUTHORIZED', '用户名或密码错误'))).toBe('用户名或密码错误')
  })

  it('refreshes an expired access token once and retries the request', async () => {
    const responses = [
      new Response(JSON.stringify({ error: { code: 'UNAUTHORIZED', message: 'token expired' } }), { status: 401 }),
      new Response(JSON.stringify({ data: { access_token: 'new-access', refresh_token: 'new-refresh', token_type: 'bearer', expires_in: 900 } }), { status: 200 }),
      new Response(JSON.stringify({ data: { id: 'u-1' } }), { status: 200 }),
    ]
    const fetchMock = vi.fn(async () => responses.shift()!)
    vi.stubGlobal('fetch', fetchMock)
    const client = new ApiClient('/api/v1', fetchMock)
    client.setSession({ accessToken: 'expired', refreshToken: 'refresh' })

    await expect(client.get<{ id: string }>('/auth/me')).resolves.toEqual({ id: 'u-1' })
    expect(client.getSession()?.accessToken).toBe('new-access')
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })
})
