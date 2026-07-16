import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { api } from '../api/client'
import type { LawyerSessionDetail, LawyerSessionItem, ReportDraft, SessionListItem, SessionState, User } from '../api/types'
import { LawyerWorkspace } from './LawyerWorkspace'

const lawyer: User = {
  id: 'lawyer-1',
  username: 'lawyer',
  real_name: '测试律师',
  role: 'lawyer',
  is_active: true,
  created_at: '2026-07-16T00:00:00Z',
}

const dbSession: LawyerSessionItem = {
  id: 'consultation-1',
  client_id: 'client-1',
  client_real_name: '张某',
  user_type: 'suspect',
  status: 'awaiting_review',
  risk_level: 'medium',
  alert_triggered: false,
  lawyer_review_needed: true,
  created_at: '2026-07-16T00:00:00Z',
  updated_at: '2026-07-16T00:00:00Z',
}

const workflowSession: SessionListItem = {
  session_id: 'workflow-1',
  consultation_id: dbSession.id,
  user_id: 'client-1',
  user_type: 'suspect',
  current_agent: 'HumanReview',
  status: 'active',
  consent_given: true,
  alert_triggered: false,
  awaiting_lawyer_review: true,
  risk_level: 'medium',
  created_at: '2026-07-16T00:00:00Z',
  updated_at: '2026-07-16T00:00:00Z',
}

const detail: LawyerSessionDetail = {
  ...dbSession,
  consent_given: true,
  report_draft: 'SQLite 报告草案',
  facts_structured: { incident_location: '杭州' },
}

const workflowState: SessionState = {
  session_id: workflowSession.session_id,
  consultation_id: dbSession.id,
  user_id: 'client-1',
  user_type: 'suspect',
  consent_given: true,
  current_agent: 'HumanReview',
  conversation_history: [],
  facts_raw: [],
  facts_structured: { incident_location: '杭州' },
  applied_laws: [],
  pending_questions: [],
  alert_triggered: false,
  risk_assessment: { risk_level: 'medium' },
  lawyer_id: lawyer.id,
  status: 'active',
}

const workflowDraft: ReportDraft = {
  session_id: workflowSession.session_id,
  consultation_id: dbSession.id,
  report_draft: '实时 workflow 报告草案',
  service_plan: { immediate_actions: ['固定证据'] },
  awaiting_lawyer_review: true,
}

function mockQueueRequests(activeSessions: SessionListItem[]) {
  return vi.spyOn(api, 'get').mockImplementation(async (path: string) => {
    if (path.startsWith('/lawyer/sessions?')) return { sessions: [dbSession], total: 1 }
    if (path === '/sessions') return { sessions: activeSessions, total: activeSessions.length }
    if (path === `/lawyer/sessions/${dbSession.id}`) return detail
    if (path === `/sessions/${workflowSession.session_id}/state`) return workflowState
    if (path === `/sessions/${workflowSession.session_id}/report-draft`) return workflowDraft
    throw new Error(`unexpected GET ${path}`)
  })
}

describe('LawyerWorkspace workflow review', () => {
  it('uses the associated workflow session for the lawyer primary approval action', async () => {
    mockQueueRequests([workflowSession])
    const putMock = vi.spyOn(api, 'put').mockResolvedValue({})
    const user = userEvent.setup()
    render(<LawyerWorkspace user={lawyer} />)

    await user.click(await screen.findByText('张某'))
    expect(await screen.findByDisplayValue('实时 workflow 报告草案')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '批准报告' }))

    await waitFor(() => expect(putMock).toHaveBeenCalledWith('/sessions/workflow-1/review', {
      decision: 'approved',
      feedback: null,
      final_output: '实时 workflow 报告草案',
    }))
    expect(putMock).not.toHaveBeenCalledWith('/lawyer/sessions/consultation-1/report', expect.anything())
  })

  it('keeps a DB-only assigned record visible but disables workflow review', async () => {
    mockQueueRequests([])
    const putMock = vi.spyOn(api, 'put').mockResolvedValue({})
    const user = userEvent.setup()
    render(<LawyerWorkspace user={lawyer} />)

    await user.click(await screen.findByText('张某'))
    expect(await screen.findByText('未关联可恢复的 active workflow')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '批准报告' })).toBeDisabled()
    expect(putMock).not.toHaveBeenCalled()
  })
})
