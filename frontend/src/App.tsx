import { useEffect, useState } from 'react'
import { api } from './api/client'
import type { User } from './api/types'
import { AuthScreen } from './auth/AuthScreen'
import { AppShell } from './components/AppShell'
import { ClientWorkspace } from './pages/ClientWorkspace'
import { LawyerWorkspace } from './pages/LawyerWorkspace'

export function App() {
  const [user, setUser] = useState<User | null>(null)
  const [restoring, setRestoring] = useState(Boolean(api.getSession()))
  const [expired, setExpired] = useState(false)

  useEffect(() => {
    api.setSessionExpiredHandler(() => { setUser(null); setExpired(true) })
    if (!api.getSession()) return
    api.get<User>('/auth/me').then(setUser).catch(() => { api.clearSession(); setExpired(true) }).finally(() => setRestoring(false))
  }, [])

  async function logout() {
    try { await api.post('/auth/logout') } catch { /* 本地会话仍需立即清理。 */ }
    api.clearSession(); setUser(null); setExpired(false)
  }

  if (restoring) return <div className="app-loading"><span className="brand-spinner" />正在恢复工作台…</div>
  if (!user) return <AuthScreen onAuthenticated={value => { setUser(value); setExpired(false) }} expired={expired} />
  return <AppShell user={user} onLogout={() => void logout()}>{user.role === 'client' ? <ClientWorkspace /> : <LawyerWorkspace user={user} />}</AppShell>
}
