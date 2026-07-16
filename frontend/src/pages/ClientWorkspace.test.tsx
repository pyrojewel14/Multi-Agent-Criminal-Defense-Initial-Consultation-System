import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { api } from '../api/client'
import type { CreateSessionResponse, SessionState } from '../api/types'
import { ClientWorkspace } from './ClientWorkspace'

const created: CreateSessionResponse = {
  session_id: 'workflow-client-1',
  consultation_id: 'consultation-client-1',
  welcome_message: '请阅读并确认知情同意。',
  current_agent: 'Receptionist',
  created_at: '2026-07-16T00:00:00Z',
}

const initialState: SessionState = {
  session_id: created.session_id,
  consultation_id: created.consultation_id,
  user_id: 'client-1',
  user_type: 'suspect',
  consent_given: false,
  current_agent: 'Receptionist',
  conversation_history: [],
  facts_raw: [],
  facts_structured: null,
  applied_laws: [],
  pending_questions: [],
  alert_triggered: false,
  status: 'active',
}

describe('ClientWorkspace core requests', () => {
  it('creates a workflow session and renders both backend identifiers', async () => {
    vi.spyOn(api, 'get').mockResolvedValue({ sessions: [], total: 0 })
    const postMock = vi.spyOn(api, 'post').mockResolvedValue(created)
    const user = userEvent.setup()
    render(<ClientWorkspace />)

    await user.click(screen.getByRole('button', { name: '创建咨询会话' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith('/sessions', { user_type: 'suspect', source: 'frontend-mvp' }))
    expect(await screen.findByText(created.session_id)).toBeInTheDocument()
    expect(screen.getByText(created.consultation_id)).toBeInTheDocument()
  })

  it('posts consent to the workflow session endpoint', async () => {
    vi.spyOn(api, 'get').mockImplementation(async (path: string) => {
      if (path === '/sessions') return { sessions: [], total: 0 }
      if (path === `/sessions/${created.session_id}/state`) return initialState
      throw new Error(`unexpected GET ${path}`)
    })
    const postMock = vi.spyOn(api, 'post').mockImplementation(async (path: string) => {
      if (path === '/sessions') return created
      if (path === `/sessions/${created.session_id}/confirm-consent`) return { next_prompt: '请描述案情', current_agent: 'FactDigger' }
      throw new Error(`unexpected POST ${path}`)
    })
    const user = userEvent.setup()
    render(<ClientWorkspace />)

    await user.click(screen.getByRole('button', { name: '创建咨询会话' }))
    await user.click(await screen.findByRole('button', { name: '确认并继续' }))

    await waitFor(() => expect(postMock).toHaveBeenCalledWith(`/sessions/${created.session_id}/confirm-consent`, expect.objectContaining({
      session_id: created.session_id,
      consent_given: true,
      consent_version: 'v1',
    })))
  })
})
