import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { ClipboardCheck, RefreshCw, UserPlus, Users } from 'lucide-react'
import { api } from '../api/client'
import type { User } from '../api/types'
import { ErrorNotice, StatusBadge } from '../components/Common'
import { LawyerWorkspace } from './LawyerWorkspace'

interface ConsultationRecord {
  id: string
  workflow_session_id?: string | null
  client_username?: string | null
  client_real_name?: string | null
  user_type: string
  status: string
  assigned_lawyer_id?: string | null
  assigned_lawyer_name?: string | null
  created_at: string
}

interface ConsultationListResponse {
  consultations: ConsultationRecord[]
  total: number
  page: number
  page_size: number
}

interface LawyerListResponse {
  lawyers: User[]
  total: number
}

type AdminTab = 'assign' | 'lawyers' | 'review'
const PAGE_SIZE = 20

export function AdminWorkspace({ user }: { user: User }) {
  const [tab, setTab] = useState<AdminTab>('assign')
  const [page, setPage] = useState(1)
  const [consultations, setConsultations] = useState<ConsultationRecord[]>([])
  const [total, setTotal] = useState(0)
  const [lawyers, setLawyers] = useState<User[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [lawyerId, setLawyerId] = useState('')
  const [username, setUsername] = useState('')
  const [realName, setRealName] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setBusy('refresh')
    setError(null)
    try {
      const [records, roster] = await Promise.all([
        api.get<ConsultationListResponse>('/consultations/list?page=' + page + '&page_size=' + PAGE_SIZE),
        api.get<LawyerListResponse>('/lawyers/'),
      ])
      setConsultations(records.consultations)
      setTotal(records.total)
      setLawyers(roster.lawyers)
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }, [page])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const selected = consultations.find(item => item.id === selectedId)
  const activeLawyers = lawyers.filter(item => item.is_active)

  function selectConsultation(item: ConsultationRecord) {
    setSelectedId(item.id)
    setLawyerId(item.assigned_lawyer_id || '')
    setNotice('')
    setError(null)
  }

  async function assignLawyer() {
    if (!selected || !lawyerId) return
    setBusy('assign')
    setError(null)
    setNotice('')
    try {
      await api.post('/consultations/assign', {
        consultation_id: selected.id,
        lawyer_id: lawyerId,
      })
      setNotice('案件已分配。律师登录后可在工作台查看此案件。')
      await refresh()
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }

  async function createLawyer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!username.trim() || password.length < 6) return
    setBusy('create')
    setError(null)
    setNotice('')
    try {
      await api.post<User>('/lawyers/', {
        username: username.trim(),
        password,
        real_name: realName.trim() || null,
      })
      setNotice('律师账号“' + username.trim() + '”已创建，可以使用设置的密码登录。')
      setUsername('')
      setRealName('')
      setPassword('')
      await refresh()
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }

  return (
    <main className="admin-workspace">
      <header className="admin-heading">
        <div>
          <p className="eyebrow">Admin workspace</p>
          <h1>咨询管理</h1>
          <p className="muted">创建律师账号、分配咨询、审核待确认的报告，全部在页面内完成。</p>
        </div>
        {tab !== 'review' && (
          <button className="secondary-button" type="button" onClick={() => void refresh()} disabled={busy !== null}>
            <RefreshCw size={16} className={busy === 'refresh' ? 'spin' : ''} />刷新
          </button>
        )}
      </header>

      <nav className="admin-tabs" aria-label="管理员功能">
        <button type="button" className={tab === 'assign' ? 'selected' : ''} onClick={() => { setTab('assign'); setError(null); setNotice('') }}>
          <ClipboardCheck size={17} />案件分配
        </button>
        <button type="button" className={tab === 'lawyers' ? 'selected' : ''} onClick={() => { setTab('lawyers'); setError(null); setNotice('') }}>
          <Users size={17} />律师账号
        </button>
        <button type="button" className={tab === 'review' ? 'selected' : ''} onClick={() => { setTab('review'); setError(null); setNotice('') }}>
          <ClipboardCheck size={17} />审核队列
        </button>
      </nav>

      {tab === 'review' ? (
        <LawyerWorkspace user={user} />
      ) : (
        <div className="admin-body">
          <ErrorNotice error={error} onRetry={() => void refresh()} />
          {notice && <p className="admin-success" role="status">{notice}</p>}

          {tab === 'assign' ? (
            <div className="admin-case-grid">
              <section className="admin-panel">
                <div className="admin-panel-header">
                  <h2>咨询记录</h2>
                  <span className="muted">共 {total} 条</span>
                </div>
                <p className="admin-hint">用户创建咨询后，就会出现在这里。选择记录即可分配承办律师。</p>
                <div className="admin-case-list">
                  {consultations.length === 0 && <p className="muted">当前没有咨询记录；请先使用当事人账号创建咨询。</p>}
                  {consultations.map(item => (
                    <button
                      type="button"
                      className={'admin-case-item' + (selectedId === item.id ? ' selected' : '')}
                      key={item.id}
                      onClick={() => selectConsultation(item)}
                    >
                      <span>{item.client_real_name || item.client_username || '未填写姓名'}</span>
                      <small>{item.status} · {item.assigned_lawyer_name || '尚未分配律师'}</small>
                      <code>{item.id.slice(0, 12)}…</code>
                    </button>
                  ))}
                </div>
                <div className="admin-pagination">
                  <button type="button" className="secondary-button" disabled={page <= 1 || busy !== null} onClick={() => { setSelectedId(null); setPage(current => current - 1) }}>上一页</button>
                  <span>第 {page} 页</span>
                  <button type="button" className="secondary-button" disabled={page * PAGE_SIZE >= total || busy !== null} onClick={() => { setSelectedId(null); setPage(current => current + 1) }}>下一页</button>
                </div>
              </section>

              <section className="admin-panel" aria-label="分配承办律师">
                <div className="admin-panel-header">
                  <h2>分配承办律师</h2>
                  {selected && <StatusBadge>{selected.status}</StatusBadge>}
                </div>
                {selected ? (
                  <div className="admin-assignment">
                    <p><strong>当事人：</strong>{selected.client_real_name || selected.client_username || '未填写姓名'}</p>
                    <p><strong>当前律师：</strong>{selected.assigned_lawyer_name || '未分配'}</p>
                    <p><strong>咨询编号：</strong><code>{selected.id}</code></p>
                    <p><strong>工作流：</strong><code>{selected.workflow_session_id || '尚无关联工作流'}</code></p>
                    <label htmlFor="admin-lawyer-select">选择律师</label>
                    <select
                      id="admin-lawyer-select"
                      value={lawyerId}
                      onChange={event => setLawyerId(event.target.value)}
                    >
                      <option value="">请选择律师</option>
                      {activeLawyers.map(lawyer => (
                        <option key={lawyer.id} value={lawyer.id}>{lawyer.real_name || lawyer.username}（{lawyer.username}）</option>
                      ))}
                    </select>
                    {activeLawyers.length === 0 && (
                      <p className="admin-hint">还没有可用律师账号，请先切换到“律师账号”创建。</p>
                    )}
                    <button type="button" className="primary-button" disabled={!lawyerId || busy !== null} onClick={() => void assignLawyer()}>
                      {busy === 'assign' ? '分配中…' : selected.assigned_lawyer_id ? '更新律师分配' : '确认分配'}
                    </button>
                    <p className="admin-hint">只有已分配律师才能访问本案状态和审核草案。若 Agent 尚未生成草案，律师审核队列会暂时为空。</p>
                  </div>
                ) : (
                  <p className="muted">从左侧选择一条咨询记录。</p>
                )}
              </section>
            </div>
          ) : (
            <div className="admin-case-grid">
              <section className="admin-panel">
                <div className="admin-panel-header"><h2>新增律师账号</h2><UserPlus size={19} /></div>
                <form className="admin-lawyer-form" onSubmit={event => void createLawyer(event)}>
                  <label htmlFor="admin-lawyer-username">登录用户名</label>
                  <input id="admin-lawyer-username" required minLength={3} maxLength={50} value={username} onChange={event => setUsername(event.target.value)} autoComplete="off" />
                  <label htmlFor="admin-lawyer-name">律师姓名（选填）</label>
                  <input id="admin-lawyer-name" value={realName} onChange={event => setRealName(event.target.value)} autoComplete="off" />
                  <label htmlFor="admin-lawyer-password">登录密码（至少 6 位）</label>
                  <input id="admin-lawyer-password" type="password" required minLength={6} maxLength={100} value={password} onChange={event => setPassword(event.target.value)} autoComplete="new-password" />
                  <button type="submit" className="primary-button" disabled={busy !== null || username.trim().length < 3 || password.length < 6}>
                    {busy === 'create' ? '创建中…' : '创建律师账号'}
                  </button>
                </form>
                <p className="admin-hint">这里只创建 lawyer 角色，普通用户不能自行注册成律师或管理员。请妥善保存并安全传递初始密码。</p>
              </section>

              <section className="admin-panel">
                <div className="admin-panel-header"><h2>现有律师</h2><span className="muted">{lawyers.length} 人</span></div>
                {lawyers.length === 0 && <p className="muted">尚未创建律师账号。</p>}
                <div className="admin-roster">
                  {lawyers.map(lawyer => (
                    <div key={lawyer.id} className="admin-roster-item">
                      <div><strong>{lawyer.real_name || lawyer.username}</strong><small>{lawyer.username}</small></div>
                      <StatusBadge tone={lawyer.is_active ? 'good' : 'warn'}>{lawyer.is_active ? '可用' : '已停用'}</StatusBadge>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}
        </div>
      )}
    </main>
  )
}
