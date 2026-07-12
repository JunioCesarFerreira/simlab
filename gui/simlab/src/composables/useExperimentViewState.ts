import { ref, watch } from 'vue'

interface ExperimentViewState {
  chartView: '2d' | '3d'
  paretoScope: 'all' | 'last'
  paretoH: number
  hvgdH: number
  evolutionH: number
  parallelH: number
  paretoXKey: string
  paretoYKey: string
  pareto3dXKey: string
  pareto3dYKey: string
  pareto3dZKey: string
}

const DEFAULTS: ExperimentViewState = {
  chartView: '2d',
  paretoScope: 'all',
  paretoH: 420,
  hvgdH: 280,
  evolutionH: 380,
  parallelH: 320,
  paretoXKey: '',
  paretoYKey: '',
  pareto3dXKey: '',
  pareto3dYKey: '',
  pareto3dZKey: '',
}

function storageKey(id: string) {
  return `simlab:exp-view:${id}`
}

export function useExperimentViewState(experimentId: string) {
  const saved = (() => {
    try {
      const raw = sessionStorage.getItem(storageKey(experimentId))
      return raw
        ? { ...DEFAULTS, ...(JSON.parse(raw) as Partial<ExperimentViewState>) }
        : { ...DEFAULTS }
    } catch {
      return { ...DEFAULTS }
    }
  })()

  const state = ref<ExperimentViewState>(saved)

  // Debounced: a resize drag mutates the state dozens of times per second;
  // one write shortly after the burst settles is enough.
  let saveTimer: ReturnType<typeof setTimeout> | null = null
  watch(
    state,
    (s) => {
      if (saveTimer) clearTimeout(saveTimer)
      saveTimer = setTimeout(() => {
        try {
          sessionStorage.setItem(storageKey(experimentId), JSON.stringify(s))
        } catch {
          // ignore quota / security errors
        }
      }, 300)
    },
    { deep: true },
  )

  return state
}
