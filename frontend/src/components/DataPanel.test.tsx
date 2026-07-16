import { render, screen } from '@testing-library/react'
import { DataPanel } from './DataPanel'

describe('DataPanel', () => {
  it('shows an honest empty state when workflow data is absent', () => {
    render(<DataPanel title="候选法条" value={[]} empty="尚未形成候选法条" />)
    expect(screen.getByText('候选法条')).toBeInTheDocument()
    expect(screen.getByText('尚未形成候选法条')).toBeInTheDocument()
  })

  it('renders structured values without collapsing nested fields', () => {
    render(<DataPanel title="结构化事实" value={{ incident_location: '杭州', evidence: ['监控'] }} />)
    expect(screen.getByText('incident_location')).toBeInTheDocument()
    expect(screen.getByText('杭州')).toBeInTheDocument()
    expect(screen.getByText('监控')).toBeInTheDocument()
  })
})
