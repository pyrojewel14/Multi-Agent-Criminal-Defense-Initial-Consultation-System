import type { ReactNode } from 'react'
import { AlertCircle, CheckCircle2, Clock3, WifiOff } from 'lucide-react'
import { getUserErrorMessage } from '../api/client'

export function ErrorNotice({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null
  return (
    <div className="notice error" role="alert">
      <AlertCircle aria-hidden="true" size={19} />
      <div><strong>操作未完成</strong><p>{getUserErrorMessage(error)}</p></div>
      {onRetry && <button className="text-button" type="button" onClick={onRetry}>重试</button>}
    </div>
  )
}

export function StatusBadge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'warn' | 'danger' }) {
  return <span className={`status-badge ${tone}`}>{children}</span>
}

export function ConnectionStatus({ online, refreshing, lastUpdated }: { online: boolean; refreshing: boolean; lastUpdated?: Date | null }) {
  return (
    <div className="connection-status" aria-live="polite">
      {!online ? <WifiOff size={16} aria-hidden="true" /> : refreshing ? <Clock3 size={16} className="spin" aria-hidden="true" /> : <CheckCircle2 size={16} aria-hidden="true" />}
      <span>{!online ? '浏览器离线' : refreshing ? '正在刷新状态' : lastUpdated ? `已刷新 ${lastUpdated.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}` : '等待首次刷新'}</span>
    </div>
  )
}

export function IdPair({ sessionId, consultationId }: { sessionId?: string | null; consultationId?: string | null }) {
  return (
    <div className="id-pair">
      <div><span>工作流 session_id</span><code title={sessionId ?? ''}>{sessionId || '尚未创建'}</code></div>
      <div><span>数据库 consultation_id</span><code title={consultationId ?? ''}>{consultationId || '尚未落库/未返回'}</code></div>
    </div>
  )
}
