import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { api } from '../api/client'
import type { User } from '../api/types'
import { AdminWorkspace } from './AdminWorkspace'

const admin: User = {
  id: 'admin-1',
  username: 'administrator',
  real_name: '演示管理员',
  role: 'admin',
  is_active: true,
  created_at: '2026-10-16T00:00:00Z',
}

const lawyer: User = {
  id: 'lawyer-1',
  username: 'demo-lawyer',
  real_name: '演示律师',
  role: 'lawyer',
  is_active: true,
  created_at: '2026-10-16T00:00:00Z',
}

const consultation = {
  id: 'consultation-1',
  workflow_session_id: 'workflow-1',
  client_username: 'test-client',
  client_real_name: '张某',
  assigned_lawyer_id: null,
  assigned_lawyer_name: null,
  user_type: 'suspect',
  status: 'in_progress',
  created_at: '2026-10-16T00:00:00Z',
}

function mockLists() {
  return vi.spyOn(api, 'get').mockImplementation(async (path: string) => {
    if (path === '/consultations/list?page=1&page_size=20') {
      return { consultations: [consultation], total: 1, page: 1, page_size: 20 }
    }
    if (path === '/lawyers/') return { lawyers: [lawyer], total: 1 }
    throw new Error('unexpected GET ' + path)
  })
}

afterEach(() => vi.restoreAllMocks())

describe('AdminWorkspace basic operations', () => {
  it('assigns a consultation to a lawyer by clicking buttons instead of hand-written API requests', async () => {
    mockLists()
    const post = vi.spyOn(api, 'post').mockResolvedValue({})
    const user = userEvent.setup()

    render(<AdminWorkspace user={admin} />)
    await user.click(await screen.findByRole('button', { name: /张某/ }))
    await user.selectOptions(screen.getByLabelText('选择律师'), 'lawyer-1')
    await user.click(screen.getByRole('button', { name: '确认分配' }))

    await waitFor(() => expect(post).toHaveBeenCalledWith('/consultations/assign', {
      consultation_id: 'consultation-1',
      lawyer_id: 'lawyer-1',
    }))
    expect(await screen.findByRole('status')).toHaveTextContent('案件已分配')
  })

  it('creates a lawyer with the protected admin endpoint and does not display the password afterwards', async () => {
    mockLists()
    const post = vi.spyOn(api, 'post').mockResolvedValue(lawyer)
    const user = userEvent.setup()

    render(<AdminWorkspace user={admin} />)
    await user.click(screen.getByRole('button', { name: '律师账号' }))
    await user.type(screen.getByLabelText('登录用户名'), 'new-lawyer')
    await user.type(screen.getByLabelText('律师姓名（选填）'), '新律师')
    await user.type(screen.getByLabelText('登录密码（至少 6 位）'), 'some-password')
    await user.click(screen.getByRole('button', { name: '创建律师账号' }))

    await waitFor(() => expect(post).toHaveBeenCalledWith('/lawyers/', {
      username: 'new-lawyer',
      password: 'some-password',
      real_name: '新律师',
    }))
    expect(await screen.findByRole('status')).toHaveTextContent('已创建')
    expect(screen.queryByDisplayValue('some-password')).not.toBeInTheDocument()
  })

  it('allows navigating to the existing administrator review queue', async () => {
    mockLists()
    const get = vi.spyOn(api, 'get')
    get.mockImplementation(async (path: string) => {
      if (path === '/sessions') return { sessions: [], total: 0 }
      if (path === '/consultations/list?page=1&page_size=20') {
        return { consultations: [consultation], total: 1, page: 1, page_size: 20 }
      }
      if (path === '/lawyers/') return { lawyers: [lawyer], total: 1 }
      throw new Error('unexpected GET ' + path)
    })
    const user = userEvent.setup()
    render(<AdminWorkspace user={admin} />)

    await user.click(screen.getByRole('button', { name: '审核队列' }))
    expect(await screen.findByRole('heading', { name: '律师审核工作台' })).toBeInTheDocument()
  })
})
