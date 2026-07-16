export type UserRole = 'client' | 'lawyer' | 'admin'

export interface User {
  id: string
  username: string
  email?: string | null
  phone?: string | null
  real_name?: string | null
  role: UserRole
  is_active: boolean
  created_at: string
  last_login_at?: string | null
}

export interface AuthTokens {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface LoginResponse extends AuthTokens {
  user: User
}

export type UserType = 'suspect' | 'victim' | 'family'

export interface CreateSessionResponse {
  session_id: string
  consultation_id: string
  welcome_message: string
  current_agent: string
  created_at: string
}

export interface SessionListItem {
  session_id: string
  consultation_id: string
  user_id: string
  user_type: string
  current_agent: string
  status: string
  consent_given: boolean
  alert_triggered: boolean
  awaiting_lawyer_review: boolean
  risk_level?: string | null
  created_at: string
  updated_at: string
}

export interface SessionState {
  session_id: string
  consultation_id?: string | null
  user_id: string
  user_type: string
  consent_given: boolean
  current_agent?: string | null
  conversation_history: unknown[]
  facts_raw: unknown[]
  facts_structured?: Record<string, unknown> | null
  applied_laws: unknown[]
  pending_questions: unknown[]
  alert_triggered: boolean
  risk_assessment?: Record<string, unknown> | null
  final_output?: string | null
  lawyer_id?: string | null
  status: string
}

export interface SendMessageResponse {
  session_id: string
  message_id: string
  agent_name: string
  response_content: string
  is_complete: boolean
  pending_questions?: unknown[] | null
  alert_triggered: boolean
  created_at: string
}

export interface LawyerSessionItem {
  id: string
  client_id: string
  client_username?: string | null
  client_real_name?: string | null
  user_type: string
  status: string
  risk_level?: string | null
  alert_triggered: boolean
  lawyer_review_needed: boolean
  created_at: string
  updated_at: string
}

export interface LawyerSessionDetail extends LawyerSessionItem {
  consent_given: boolean
  facts_raw?: string[] | null
  facts_structured?: Record<string, unknown> | null
  applied_laws?: Record<string, unknown>[] | null
  risk_assessment?: Record<string, unknown> | null
  report_draft?: string | null
  service_plan?: Record<string, unknown> | null
  final_output?: string | null
  conversation_history?: Array<{
    id: string
    sender_type: string
    content: string
    agent_name?: string | null
    message_type: string
    created_at: string
  }> | null
  completed_at?: string | null
}

export interface ReportDraft {
  session_id: string
  consultation_id?: string | null
  report_draft: string
  service_plan?: Record<string, unknown> | null
  awaiting_lawyer_review: boolean
}
