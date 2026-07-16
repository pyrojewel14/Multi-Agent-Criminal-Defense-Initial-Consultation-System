import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'

const values = new Map<string, string>()
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  value: {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
  },
})

afterEach(() => {
  window.localStorage.clear()
  vi.restoreAllMocks()
})
