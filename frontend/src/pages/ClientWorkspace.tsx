import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react'
import { ArrowUp, Check, MessageSquareText, Plus, RefreshCw, ShieldCheck } from 'lucide-react'
import { api } from '../api/client'
import type { CreateSessionResponse, SendMessageResponse, SessionListItem, SessionState, UserType } from '../api/types'
import { ConnectionStatus, ErrorNotice, IdPair, StatusBadge } from '../components/Common'
import { DataPanel } from '../components/DataPanel'

const userTypeNames: Record<UserType, string> = { suspect: '嫌疑人 / 被调查人', victim: '被害人', family: '家属' }
const agentNames: Record<string, string> = {
  Receptionist: '接待与告知', FactDigger: '事实梳理', LawRef: '候选法条检索', RiskAssessor: '风险评估', ServicePlanner: '服务计划', HumanReview: '律师审核', HumanAlert: '高风险人工介入', WaitForUser: '等待补充信息', END: '流程结束',
}

interface LocalMessage { id: string; sender: 'user' | 'agent' | 'system'; content: string; label?: string }

export function ClientWorkspace() {
  const [sessions, setSessions] = useState<SessionListItem[]>([])
  const [active, setActive] = useState<CreateSessionResponse | SessionListItem | null>(null)
  const [state, setState] = useState<SessionState | null>(null)
  const [messages, setMessages] = useState<LocalMessage[]>([])
  const [userType, setUserType] = useState<UserType>('suspect')
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [online, setOnline] = useState(navigator.onLine)

  const refreshSessions = useCallback(async () => {
    try {
      const result = await api.get<{ sessions: SessionListItem[]; total: number }>('/sessions')
      setSessions(result.sessions)
    } catch (cause) {
      setError(cause)
    }
  }, [])

  const refreshState = useCallback(async (silent = false) => {
    if (!active) return
    if (!silent) setRefreshing(true)
    try {
      const result = await api.get<SessionState>(`/sessions/${active.session_id}/state`)
      setState(result)
      setLastUpdated(new Date())
      setError(null)
    } catch (cause) {
      setError(cause)
    } finally {
      if (!silent) setRefreshing(false)
    }
  }, [active])

  useEffect(() => { void refreshSessions() }, [refreshSessions])
  useEffect(() => {
    const onOnline = () => setOnline(true)
    const onOffline = () => setOnline(false)
    window.addEventListener('online', onOnline)
    window.addEventListener('offline', onOffline)
    return () => { window.removeEventListener('online', onOnline); window.removeEventListener('offline', onOffline) }
  }, [])
  useEffect(() => {
    if (!active || !online) return
    void refreshState()
    const timer = window.setInterval(() => void refreshState(true), 8000)
    return () => window.clearInterval(timer)
  }, [active, online, refreshState])

  async function createSession() {
    setBusy('create')
    setError(null)
    try {
      const result = await api.post<CreateSessionResponse>('/sessions', { user_type: userType, source: 'frontend-mvp' })
      setActive(result)
      setMessages([{ id: 'welcome', sender: 'agent', content: result.welcome_message, label: '接待与告知' }])
      setState(null)
      await refreshSessions()
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }

  async function confirmConsent() {
    if (!active) return
    setBusy('consent')
    setError(null)
    try {
      const result = await api.post<{ next_prompt: string; current_agent: string }>(`/sessions/${active.session_id}/confirm-consent`, {
        session_id: active.session_id,
        consent_given: true,
        consent_timestamp: new Date().toISOString(),
        consent_version: 'v1',
        identity_info: { role: userType },
      })
      setMessages(current => [...current, { id: `consent-${Date.now()}`, sender: 'system', content: '已记录知情同意（v1）' }, { id: `prompt-${Date.now()}`, sender: 'agent', content: result.next_prompt, label: agentNames[result.current_agent] || result.current_agent }])
      await refreshState()
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault()
    const content = draft.trim()
    if (!active || !content) return
    setBusy('message')
    setError(null)
    setMessages(current => [...current, { id: `user-${Date.now()}`, sender: 'user', content }])
    setDraft('')
    try {
      const result = await api.post<SendMessageResponse>(`/sessions/${active.session_id}/message`, { session_id: active.session_id, content, message_type: 'text' })
      if (result.response_content) setMessages(current => [...current, { id: result.message_id, sender: 'agent', content: result.response_content, label: agentNames[result.agent_name] || result.agent_name }])
      await refreshState()
    } catch (cause) {
      setError(cause)
    } finally {
      setBusy(null)
    }
  }

  const currentAgent = state?.current_agent || active?.current_agent
  const consentGiven = state?.consent_given ?? false
  const workflowCompleted = state?.status === 'completed'
  const selectedIds = useMemo(() => ({ sessionId: active?.session_id, consultationId: active?.consultation_id || state?.consultation_id }), [active, state])

  return (
    <main className="workspace client-workspace">
      <aside className="session-rail">
        <div className="rail-heading"><div><p className="eyebrow">Client workspace</p><h1>我的咨询</h1></div><button className="icon-button solid" type="button" onClick={() => { setActive(null); setState(null); setMessages([]) }} aria-label="新建咨询"><Plus size={18} /></button></div>
        <div className="session-list">
          {sessions.length === 0 && <div className="empty-state rail-empty"><MessageSquareText size={20} /><p>尚无运行中的咨询</p></div>}
          {sessions.map(session => (
            <button className={`session-item ${active?.session_id === session.session_id ? 'active' : ''}`} type="button" key={session.session_id} onClick={() => { setActive(session); setState(null); setMessages([]) }}>
              <span>{userTypeNames[session.user_type as UserType] || session.user_type}</span>
              <small>{agentNames[session.current_agent] || session.current_agent}</small>
              <code>{session.session_id.slice(0, 8)}…</code>
            </button>
          ))}
        </div>
        <div className="rail-note"><ShieldCheck size={17} /><p>自动整理不构成正式法律意见；高风险内容会停止自动流程并提示人工处理。</p></div>
      </aside>

      <section className="workspace-main">
        {!active ? (
          <div className="create-session-view">
            <div><p className="eyebrow">开始一次新的初步整理</p><h2>请选择您在案件中的身份</h2><p>创建后会先显示权利义务告知。确认知情同意前，系统不会进入事实收集。</p></div>
            <div className="identity-options">
              {(Object.entries(userTypeNames) as Array<[UserType, string]>).map(([value, label]) => <button key={value} className={userType === value ? 'selected' : ''} onClick={() => setUserType(value)} type="button"><span>{label}</span><small>{value === 'suspect' ? '本人正被调查或可能承担刑事责任' : value === 'victim' ? '本人权益可能因案件受到侵害' : '为亲属了解初步情况'}</small></button>)}
            </div>
            <ErrorNotice error={error} />
            <button className="primary-button" type="button" disabled={busy === 'create'} onClick={createSession}>{busy === 'create' ? '正在创建…' : '创建咨询会话'}<Plus size={17} /></button>
          </div>
        ) : (
          <>
            <header className="workspace-header">
              <div><p className="eyebrow">当前会话</p><h2>{userTypeNames[(state?.user_type || userType) as UserType] || '刑事初次咨询'}</h2><div className="status-line"><StatusBadge tone={state?.alert_triggered ? 'danger' : workflowCompleted || consentGiven ? 'good' : 'warn'}>{state?.alert_triggered ? '已转人工' : workflowCompleted ? '律师已审核' : consentGiven ? '已同意' : '等待知情同意'}</StatusBadge><StatusBadge>{workflowCompleted ? '流程完成' : agentNames[currentAgent || ''] || currentAgent || '状态加载中'}</StatusBadge></div></div>
              <div className="header-tools"><ConnectionStatus online={online} refreshing={refreshing} lastUpdated={lastUpdated} /><button className="secondary-button" type="button" onClick={() => void refreshState()} disabled={refreshing}><RefreshCw size={16} />刷新</button></div>
            </header>
            <IdPair sessionId={selectedIds.sessionId} consultationId={selectedIds.consultationId} />
            <ErrorNotice error={error} onRetry={() => void refreshState()} />
            <div className="client-grid">
              <section className="conversation-panel">
                <div className="conversation-heading"><div><p className="eyebrow">逐步补充</p><h3>咨询对话</h3></div><span>系统可能继续追问缺失事实</span></div>
                <div className="message-list" aria-live="polite">
                  {messages.length === 0 && <div className="empty-state"><MessageSquareText size={22} /><p>已恢复会话状态，但历史消息以 SQLite 记录为准；当前实时状态不一定包含完整对话。</p></div>}
                  {messages.map(message => <article className={`message ${message.sender}`} key={message.id}><div className="message-label">{message.sender === 'user' ? '我' : message.sender === 'system' ? '系统记录' : message.label || '咨询助手'}</div><p>{message.content}</p></article>)}
                </div>
                {!consentGiven ? (
                  <div className="consent-box"><ShieldCheck size={23} /><div><h4>继续前需确认</h4><p>我已阅读欢迎语中的权利义务告知，理解系统输出仅用于初步信息整理，并同意记录本次咨询内容。</p></div><button className="primary-button" type="button" disabled={busy === 'consent'} onClick={confirmConsent}>{busy === 'consent' ? '正在记录…' : '确认并继续'}<Check size={17} /></button></div>
                ) : (
                  <form className="composer" onSubmit={sendMessage}>{workflowCompleted && <p className="composer-status">咨询流程已完成，新的案情请创建新咨询。</p>}<label htmlFor="consultation-message" className="sr-only">补充案情</label><textarea id="consultation-message" rows={3} value={draft} onChange={event => setDraft(event.target.value)} placeholder="请按时间顺序描述经过、地点、人员、证据和当前办案状态…" disabled={busy === 'message' || state?.alert_triggered || workflowCompleted} /><button className="send-button" type="submit" aria-label="发送消息" disabled={!draft.trim() || busy === 'message' || state?.alert_triggered || workflowCompleted}>{busy === 'message' ? <RefreshCw className="spin" size={19} /> : <ArrowUp size={19} />}</button></form>
                )}
              </section>
              <section className="insight-stack">
                <DataPanel title="待补充问题" eyebrow="Next questions" value={state?.pending_questions || []} empty="当前没有待补充问题" />
                <DataPanel title="结构化事实" eyebrow="Facts" value={state?.facts_structured} empty="完成事实收集后显示结构化字段" />
                <DataPanel title="候选法条" eyebrow="References" value={state?.applied_laws || []} empty="尚未形成候选法条；候选项不等于定罪结论" />
                <DataPanel title="风险评估" eyebrow="Risk" value={state?.risk_assessment} empty="事实与法条信息足够后生成初步风险提示" />
                <DataPanel title="服务计划 / 报告" eyebrow="Draft" value={currentAgent === 'HumanReview' || currentAgent === 'END' ? state?.final_output : null} empty={state?.lawyer_id ? '草案需在律师工作台审核；client 状态暂未返回可展示文本' : '尚未进入律师审核阶段，也未分配律师'} />
              </section>
            </div>
          </>
        )}
      </section>
    </main>
  )
}
