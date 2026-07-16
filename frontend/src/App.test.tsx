import { render, screen } from '@testing-library/react'
import { App } from './App'

describe('App', () => {
  it('opens directly on the authentication workspace without a marketing hero', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: '进入初询工作台' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '登录' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '注册新账号' })).toBeInTheDocument()
  })
})
