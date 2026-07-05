import { ref, watch } from 'vue'

interface ExperimentViewState {
  chartView: '2d' | '3d'
  paretoH: number
  hvgdH: number
  evolutionH: number
  paretoXKey: string
  paretoYKey: string
  pareto3dXKey: string
  pareto3dYKey: string
  pareto3dZKey: string
}

const DEFAULTS: ExperimentViewState = {
  chartView: '2d',
  paretoH: 420,
  hvgdH: 280,
  evolutionH: 380,
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

  watch(
    state,
    (s) => {
      try {
        sessionStorage.setItem(storageKey(experimentId), JSON.stringify(s))
      } catch {
        // ignore quota / security errors
      }
    },
    { deep: true },
  )

  return state
}
