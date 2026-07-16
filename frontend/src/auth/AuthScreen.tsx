import { useState, type FormEvent } from 'react'
import { ArrowRight, LockKeyhole, Scale, ShieldCheck } from 'lucide-react'
import { api } from '../api/client'
import type { LoginResponse, User } from '../api/types'
import { ErrorNotice } from '../components/Common'

export function AuthScreen({ onAuthenticated, expired }: { onAuthenticated: (user: User) => void; expired?: boolean }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [realName, setRealName] = useState('')
  const [email, setEmail] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      if (mode === 'register') {
        await api.post('/auth/register', {
          username,
          password,
          real_name: realName || null,
          email: email || null,
        }, { skipAuth: true })
      }
      const result = await api.post<LoginResponse>('/auth/login', { username, password }, { skipAuth: true })
      api.setSession({ accessToken: result.access_token, refreshToken: result.refresh_token })
      onAuthenticated(result.user)
    } catch (cause) {
      setError(cause)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="auth-layout">
      <section className="auth-context" aria-label="系统说明">
        <div className="brand-lockup"><span className="brand-mark"><Scale size={25} /></span><span>初询台</span></div>
        <div className="context-copy">
          <p className="eyebrow">刑事辩护 · 初次咨询</p>
          <h1>把零散叙述整理成<br />可供律师审核的案件线索</h1>
          <p>系统会先完成知情同意，再逐步收集事实、检索候选法条并提示风险。自动结果仅用于初步整理，不能替代律师意见。</p>
        </div>
        <div className="trust-list">
          <span><ShieldCheck size={18} />知情同意后才开始咨询</span>
          <span><LockKeyhole size={18} />JWT 会话与角色权限控制</span>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-card">
          <p className="eyebrow">{mode === 'login' ? '欢迎回来' : '创建 client 账号'}</p>
          <h2>进入初询工作台</h2>
          <p className="form-intro">{mode === 'login' ? 'client、lawyer 与 admin 使用同一入口，登录后按真实角色进入对应工作台。' : '公开注册只会创建 client 角色；律师和管理员账号需由受信任管理员配置。'}</p>
          {expired && <div className="notice warn" role="status"><AlertText /></div>}
          <ErrorNotice error={error} />
          <form onSubmit={submit} className="auth-form">
            <label>用户名<input autoComplete="username" minLength={3} maxLength={50} value={username} onChange={event => setUsername(event.target.value)} required /></label>
            {mode === 'register' && <label>真实姓名（选填）<input autoComplete="name" value={realName} onChange={event => setRealName(event.target.value)} /></label>}
            {mode === 'register' && <label>邮箱（选填）<input type="email" autoComplete="email" value={email} onChange={event => setEmail(event.target.value)} /></label>}
            <label>密码<input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} minLength={6} maxLength={100} value={password} onChange={event => setPassword(event.target.value)} required /></label>
            <button className="primary-button full" disabled={loading} type="submit">
              {loading ? '正在连接…' : mode === 'login' ? '登录' : '注册并登录'}<ArrowRight size={17} aria-hidden="true" />
            </button>
          </form>
          <button className="switch-button" type="button" disabled={loading} onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null) }}>
            {mode === 'login' ? '注册新账号' : '返回登录'}
          </button>
        </div>
      </section>
    </main>
  )
}

function AlertText() {
  return <><LockKeyhole size={18} /><div><strong>登录已过期</strong><p>刷新令牌也已失效，请重新登录。</p></div></>
}
