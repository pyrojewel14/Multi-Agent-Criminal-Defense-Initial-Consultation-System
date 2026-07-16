import type { ReactNode } from 'react'
import { FileSearch } from 'lucide-react'

interface DataPanelProps {
  title: string
  value: unknown
  empty?: string
  eyebrow?: string
  className?: string
}

function hasValue(value: unknown): boolean {
  if (value === null || value === undefined || value === '') return false
  if (Array.isArray(value)) return value.length > 0
  if (typeof value === 'object') return Object.keys(value as object).length > 0
  return true
}

function renderValue(value: unknown, keyPath = 'value'): ReactNode {
  if (value === null || value === undefined || value === '') return <span className="muted">未提供</span>
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'string' || typeof value === 'number') return <span className="data-value">{String(value)}</span>
  if (Array.isArray(value)) {
    return (
      <ul className="data-list">
        {value.map((item, index) => <li key={`${keyPath}-${index}`}>{renderValue(item, `${keyPath}-${index}`)}</li>)}
      </ul>
    )
  }
  if (typeof value === 'object') {
    return (
      <dl className="data-grid">
        {Object.entries(value as Record<string, unknown>).map(([key, item]) => (
          <div className="data-row" key={`${keyPath}-${key}`}>
            <dt>{key}</dt>
            <dd>{renderValue(item, `${keyPath}-${key}`)}</dd>
          </div>
        ))}
      </dl>
    )
  }
  return String(value)
}

export function DataPanel({ title, value, empty = '当前阶段尚无数据', eyebrow, className = '' }: DataPanelProps) {
  return (
    <section className={`data-panel ${className}`}>
      <header className="panel-heading">
        <div>
          {eyebrow && <p className="eyebrow">{eyebrow}</p>}
          <h3>{title}</h3>
        </div>
      </header>
      {hasValue(value) ? renderValue(value, title) : (
        <div className="empty-state compact">
          <FileSearch aria-hidden="true" size={20} />
          <p>{empty}</p>
        </div>
      )}
    </section>
  )
}
