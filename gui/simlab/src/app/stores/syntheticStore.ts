import { defineStore } from 'pinia'
import { ref } from 'vue'

export type BenchmarkId = 'DTLZ2' | 'ZDT1' | 'SCH1'

export interface BenchmarkDraft {
  benchmark: BenchmarkId
  M: number            // number of objectives
  nVars: number        // decision variables (n_relays = ceil(nVars/2))
  noiseStd: number
  region: [number, number, number, number]  // [xmin, ymin, xmax, ymax]
}

const STORAGE_KEY = 'simlab:synthetic-draft'

function defaultDraft(): BenchmarkDraft {
  return {
    benchmark: 'DTLZ2',
    M: 3,
    nVars: 10,
    noiseStd: 0.0,
    region: [-100, -100, 100, 100],
  }
}

function load(): BenchmarkDraft {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return defaultDraft()
    return { ...defaultDraft(), ...JSON.parse(raw) }
  } catch {
    return defaultDraft()
  }
}

function save(draft: BenchmarkDraft) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(draft))
  } catch { /* storage full — silently skip */ }
}

export const useSyntheticStore = defineStore('synthetic', () => {
  const draft = ref<BenchmarkDraft>(load())

  function setDraft(d: Partial<BenchmarkDraft>) {
    draft.value = { ...draft.value, ...d }
    // Enforce benchmark constraints
    if (draft.value.benchmark === 'ZDT1' || draft.value.benchmark === 'SCH1') {
      draft.value.M = 2
    }
    if (draft.value.benchmark === 'DTLZ2' && draft.value.nVars < draft.value.M - 1) {
      draft.value.nVars = draft.value.M - 1
    }
    if (draft.value.benchmark === 'ZDT1' && draft.value.nVars < 2) {
      draft.value.nVars = 2
    }
    if (draft.value.benchmark === 'SCH1') {
      draft.value.nVars = 1  // only x[0] is used; n is fixed at 1
    }
    save(draft.value)
  }

  function reset() {
    draft.value = defaultDraft()
    save(draft.value)
  }

  /** Number of relay motes needed to encode nVars decision variables. */
  function nRelays(): number {
    return Math.ceil(draft.value.nVars / 2)
  }

  return { draft, setDraft, reset, nRelays }
})
