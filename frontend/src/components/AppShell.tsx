import type { ReactNode } from 'react'
import { LogOut, Scale } from 'lucide-react'
import type { User } from '../api/types'
import { StatusBadge } from './Common'

const roleNames = { client: '咨询用户', lawyer: '承办律师', admin: '管理员' }

export function AppShell({ user, onLogout, children }: { user: User; onLogout: () => void; children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup compact"><span className="brand-mark"><Scale size={21} /></span><span>初询台</span></div>
        <div className="topbar-actions">
          <div className="user-meta"><span>{user.real_name || user.username}</span><StatusBadge tone={user.role === 'client' ? 'neutral' : 'good'}>{roleNames[user.role]}</StatusBadge></div>
          <button className="icon-button" type="button" onClick={onLogout} aria-label="退出登录"><LogOut size={18} /></button>
        </div>
      </header>
      {children}
    </div>
  )
}
