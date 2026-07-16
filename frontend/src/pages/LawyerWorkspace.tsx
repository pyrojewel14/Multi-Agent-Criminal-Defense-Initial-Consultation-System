import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, ClipboardCheck, RefreshCw, RotateCcw, Search } from 'lucide-react'
import { api } from '../api/client'
import type { LawyerSessionDetail, LawyerSessionItem, ReportDraft, SessionListItem, SessionState, User } from '../api/types'
import { ErrorNotice, IdPair, StatusBadge } from '../components/Common'
import { DataPanel } from '../components/DataPanel'

interface ReviewQueueItem {
  consultationId: string
  db?: LawyerSessionItem
  workflow?: SessionListItem
}

function mergeLawyerQueues(dbSessions: LawyerSessionItem[], workflowSessions: SessionListItem[]) {
  const activeByConsultation = new Map(
    workflowSessions
      .filter(item => item.awaiting_lawyer_review)
      .map(item => [item.consultation_id, item]),
  )
  const merged: ReviewQueueItem[] = dbSessions.map(db => ({
    consultationId: db.id,
    db,
    workflow: activeByConsultation.get(db.id),
  }))
  const dbIds = new Set(dbSessions.map(item => item.id))

  for (const workflow of workflowSessions) {
    if (workflow.awaiting_lawyer_review && !dbIds.has(workflow.consultation_id)) {
      merged.push({ consultationId: workflow.consultation_id, workflow })
    }
  }
  return merged
}

export function LawyerWorkspace({ user }: { user: User }) {
  const [queueItems, setQueueItems] = useState<ReviewQueueItem[]>([])
  const [selected, setSelected] = useState<ReviewQueueItem | null>(null)
  const [detail, setDetail] = useState<LawyerSessionDetail | null>(null)
  const [workflowState, setWorkflowState] = useState<SessionState | null>(null)
  const [workflowDraft, setWorkflowDraft] = useState<ReportDraft | null>(null)
  const [finalOutput, setFinalOutput] = useState('')
  const [feedback, setFeedback] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)

  const loadQueues = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      if (user.role === 'lawyer') {
        const [dbResult, workflowResult] = await Promise.all([
          api.get<{ sessions: LawyerSessionItem[]; total: number }>('/lawyer/sessions?needs_review=true&page=1&page_size=50'),
          api.get<{ sessions: SessionListItem[]; total: number }>('/sessions'),
        ])
        setQueueItems(mergeLawyerQueues(dbResult.sessions, workflowResult.sessions))
      } else {
        const result = await api.get<{ sessions: SessionListItem[]; total: number }>('/sessions')
        setQueueItems(result.sessions
          .filter(item => item.awaiting_lawyer_review)
          .map(workflow => ({ consultationId: workflow.consultation_id, workflow })))
      }
    } catch (cause) {
      setError(cause)
    } finally {
      setLoading(false)
    }
  }, [user.role])

  useEffect(() => { void loadQueues() }, [loadQueues])

  async function openQueueItem(item: ReviewQueueItem) {
    setSelected(item)
    setDetail(null)
    setWorkflowState(null)
    setWorkflowDraft(null)
    setFinalOutput('')
    setFeedback('')
    setLoading(true)
    setError(null)
    try {
      const [dbResult, stateResult, draftResult] = await Promise.all([
        item.db ? api.get<LawyerSessionDetail>(`/lawyer/sessions/${item.db.id}`) : Promise.resolve(null),
        item.workflow ? api.get<SessionState>(`/sessions/${item.workflow.session_id}/state`) : Promise.resolve(null),
        item.workflow ? api.get<ReportDraft>(`/sessions/${item.workflow.session_id}/report-draft`) : Promise.resolve(null),
      ])
      setDetail(dbResult)
      setWorkflowState(stateResult)
      setWorkflowDraft(draftResult)
      setFinalOutput(draftResult?.report_draft || dbResult?.report_draft || dbResult?.final_output || '')
    } catch (cause) {
      setError(cause)
    } finally {
      setLoading(false)
    }
  }

  function clearSelection() {
    setSelected(null)
    setDetail(null)
    setWorkflowState(null)
    setWorkflowDraft(null)
    setFinalOutput('')
    setFeedback('')
  }

  async function approve() {
    if (!selected?.workflow || !finalOutput.trim()) return
    setLoading(true)
    setError(null)
    try {
      await api.put(`/sessions/${selected.workflow.session_id}/review`, {
        decision: 'approved',
        feedback: feedback || null,
        final_output: finalOutput,
      })
      clearSelection()
      await loadQueues()
    } catch (cause) {
      setError(cause)
    } finally {
      setLoading(false)
    }
  }

  async function revise(decision: 'revise_facts' | 'revise_risk') {
    if (!selected?.workflow) return
    setLoading(true)
    setError(null)
    try {
      await api.put(`/sessions/${selected.workflow.session_id}/review`, {
        decision,
        feedback: feedback || null,
        final_output: null,
      })
      clearSelection()
      await loadQueues()
    } catch (cause) {
      setError(cause)
    } finally {
      setLoading(false)
    }
  }

  const facts = workflowState?.facts_structured || detail?.facts_structured
  const laws = workflowState?.applied_laws || detail?.applied_laws
  const risk = workflowState?.risk_assessment || detail?.risk_assessment
  const servicePlan = workflowDraft?.service_plan || detail?.service_plan
  const hasLiveWorkflow = Boolean(selected?.workflow)

  return (
    <main className="lawyer-workspace">
      <aside className="review-queue">
        <div className="rail-heading"><div><p className="eyebrow">{user.role === 'admin' ? 'Admin workflow queue' : 'Assigned lawyer queue'}</p><h1>待审核草案</h1></div><button className="icon-button solid" type="button" onClick={() => void loadQueues()} aria-label="刷新审核队列"><RefreshCw size={18} className={loading ? 'spin' : ''} /></button></div>
        <div className="permission-note"><strong>{user.role === 'admin' ? '管理员实时队列' : '分配记录与实时 workflow 联合队列'}</strong><p>{user.role === 'admin' ? '使用 workflow session_id 读取实时草案。' : '按 consultation_id 关联 SQLite 分配记录与当前账号可访问的 active workflow；只有关联成功的记录可恢复 Agent。'}</p></div>
        <div className="session-list">
          {queueItems.length === 0 && !loading && <div className="empty-state rail-empty"><ClipboardCheck size={21} /><p>{user.role === 'admin' ? '当前没有处于律师审核断点的活跃 workflow' : '当前账号没有已分配记录或待审核的活跃 workflow'}</p></div>}
          {queueItems.map(item => (
            <button key={`${item.consultationId}-${item.workflow?.session_id || 'db'}`} type="button" className={`session-item ${selected?.consultationId === item.consultationId ? 'active' : ''}`} onClick={() => void openQueueItem(item)}>
              <span>{item.db?.client_real_name || item.db?.client_username || item.workflow?.user_type || '未填写姓名'}</span>
              <small>{item.workflow ? '可恢复 workflow' : '仅 SQLite 历史 · 不可恢复'}</small>
              <code>{item.consultationId.slice(0, 8)}…</code>
            </button>
          ))}
        </div>
      </aside>

      <section className="review-main">
        <header className="workspace-header"><div><p className="eyebrow">Human in the loop</p><h2>{selected ? '审核初步咨询草案' : '律师审核工作台'}</h2><div className="status-line"><StatusBadge tone="warn">人工审核门禁</StatusBadge>{selected && <StatusBadge tone={hasLiveWorkflow ? 'good' : 'warn'}>{hasLiveWorkflow ? '实时 workflow 可恢复' : '仅数据库记录'}</StatusBadge>}</div></div></header>
        <ErrorNotice error={error} onRetry={() => selected ? void openQueueItem(selected) : void loadQueues()} />
        {!selected ? <div className="review-empty"><Search size={32} /><h3>从左侧选择一条待审核记录</h3><p>如果队列为空，表示当前账号没有可访问的分配或活跃草案。系统不会生成虚构案件供演示。</p></div> : (
          <>
            <IdPair sessionId={selected.workflow?.session_id} consultationId={selected.consultationId} />
            {!hasLiveWorkflow && <div className="notice warn" role="status"><AlertTriangle size={18} /><div><strong>未关联可恢复的 active workflow</strong><p>此项仅保留 SQLite 咨询记录作为持久化证据。数据库报告接口不会恢复 Agent workflow，因此主要审核动作已禁用。</p></div></div>}
            <div className="review-grid">
              <div className="review-evidence"><DataPanel title="结构化事实" value={facts} empty="当前记录没有已持久化或实时结构化事实" /><DataPanel title="候选法条" value={laws} empty="当前记录没有候选法条" /><DataPanel title="风险评估" value={risk} empty="当前记录没有风险评估" /><DataPanel title="服务计划" value={servicePlan} empty="当前记录没有服务计划" /></div>
              <section className="draft-editor"><div><p className="eyebrow">Review draft</p><h3>报告草案与审核意见</h3></div><label>最终报告<textarea rows={14} value={finalOutput} readOnly={!hasLiveWorkflow} onChange={event => setFinalOutput(event.target.value)} placeholder="当前没有可审核的报告草案" /></label><label>审核反馈（选填）<textarea rows={4} value={feedback} disabled={!hasLiveWorkflow} onChange={event => setFeedback(event.target.value)} placeholder="记录核对结论或需修改的具体内容" /></label><div className="review-actions"><button type="button" className="secondary-button" disabled={loading || !hasLiveWorkflow} onClick={() => void revise('revise_facts')}><RotateCcw size={16} />退回事实</button><button type="button" className="secondary-button" disabled={loading || !hasLiveWorkflow} onClick={() => void revise('revise_risk')}><RotateCcw size={16} />重评风险</button><button type="button" className="primary-button" disabled={loading || !hasLiveWorkflow || !finalOutput.trim()} onClick={() => void approve()}><ClipboardCheck size={17} />批准报告</button></div></section>
            </div>
          </>
        )}
      </section>
    </main>
  )
}
