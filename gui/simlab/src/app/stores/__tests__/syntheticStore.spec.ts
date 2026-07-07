import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useSyntheticStore } from '../syntheticStore'

// In-memory localStorage stub (the store persists the draft on every change)
function stubLocalStorage() {
  const mem: Record<string, string> = {}
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => (k in mem ? mem[k] : null),
    setItem: (k: string, v: string) => { mem[k] = v },
    removeItem: (k: string) => { delete mem[k] },
    clear: () => { for (const k of Object.keys(mem)) delete mem[k] },
  })
}

beforeEach(() => {
  stubLocalStorage()
  setActivePinia(createPinia())
})

describe('syntheticStore', () => {
  it('starts from the DTLZ2 default draft', () => {
    const store = useSyntheticStore()
    expect(store.draft.benchmark).toBe('DTLZ2')
    expect(store.draft.M).toBe(3)
    expect(store.draft.nVars).toBe(10)
  })

  it('forces M=2 when switching to ZDT1', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'ZDT1', M: 5 })
    expect(store.draft.M).toBe(2)
  })

  it('forces M=2 when switching to SCH1', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'SCH1', M: 4 })
    expect(store.draft.M).toBe(2)
  })

  it('raises nVars to M-1 for DTLZ2 when too small', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'DTLZ2', M: 5, nVars: 2 })
    expect(store.draft.nVars).toBe(4) // M - 1
  })

  it('enforces minimum nVars per benchmark', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'ZDT1', nVars: 1 })
    expect(store.draft.nVars).toBe(2)
    store.setDraft({ benchmark: 'SCH1', nVars: 0 })
    expect(store.draft.nVars).toBe(1)
  })

  it('computes nRelays as ceil(nVars/2)', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'DTLZ2', M: 2, nVars: 7 })
    expect(store.nRelays()).toBe(4)
    store.setDraft({ nVars: 8 })
    expect(store.nRelays()).toBe(4)
  })

  it('reset() restores the default draft', () => {
    const store = useSyntheticStore()
    store.setDraft({ benchmark: 'ZDT1', nVars: 20 })
    store.reset()
    expect(store.draft).toEqual({
      benchmark: 'DTLZ2',
      M: 3,
      nVars: 10,
      noiseStd: 0,
      region: [-100, -100, 100, 100],
    })
  })

  it('persists the draft to localStorage', () => {
    const store = useSyntheticStore()
    store.setDraft({ nVars: 12 })
    const raw = localStorage.getItem('simlab:synthetic-draft')
    expect(raw).toBeTruthy()
    expect(JSON.parse(raw as string).nVars).toBe(12)
  })
})
