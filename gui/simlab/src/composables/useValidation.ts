import { computed } from 'vue'
import { useProblemStore } from '../app/stores/problemStore'
import { validateProblem, validateChromosome } from '../services/validators'

export function useValidation() {
  const problemStore = useProblemStore()
  const errors = computed(() => validateProblem(problemStore.draft))
  const chromosomeErrors = computed(() => validateChromosome(problemStore.draft))
  const errorFields = computed(() => new Set([
    ...errors.value.map(e => e.field),
    ...chromosomeErrors.value.map(e => e.field),
  ]))

  function hasError(field: string) {
    return errorFields.value.has(field)
  }

  function errorFor(field: string): string | null {
    return (
      errors.value.find(e => e.field === field)?.message ??
      chromosomeErrors.value.find(e => e.field === field)?.message ??
      null
    )
  }

  return { errors, chromosomeErrors, hasError, errorFor }
}
