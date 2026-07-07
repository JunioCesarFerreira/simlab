<template>
  <Teleport to="body">
    <div class="backdrop">
      <div
class="modal" role="dialog" aria-modal="true"
        :aria-label="`Launch synthetic experiment — step ${currentStep} of ${TOTAL_STEPS}`">

        <!-- Header -->
        <div class="modal-header">
          <div class="header-title">
            <span class="header-icon">⬡</span>
            <div>
              <h2 class="title">Launch synthetic experiment</h2>
              <p class="subtitle">{{ STEP_LABELS[currentStep - 1] }}</p>
            </div>
          </div>
          <button class="close-btn" aria-label="Close" @click="requestClose">✕</button>
        </div>

        <!-- Step indicator -->
        <div class="step-indicator" aria-hidden="true">
          <div
            v-for="s in TOTAL_STEPS" :key="s"
            class="step-dot"
            :class="{ 'step-dot--done': s < currentStep, 'step-dot--active': s === currentStep }"
            @click="goToStep(s)"
          >
            <span class="step-dot-num">{{ s < currentStep ? '✓' : s }}</span>
            <span class="step-dot-label">{{ STEP_SHORT[s - 1] }}</span>
          </div>
          <div class="step-line" />
        </div>

        <!-- Body -->
        <div class="modal-body">
          <SLStep1Review v-if="currentStep === 1" />
          <SLStep2Algorithm
            v-else-if="currentStep === 2"
            v-model="form.algorithm"
            :show-validation="showValidation"
          />
          <SLStep3Objectives
            v-else-if="currentStep === 3"
            v-model="form.objectives"
            :show-validation="showValidation"
            :objective-count="store.draft.M"
          />
          <SLStep4Confirm
            v-else-if="currentStep === 4"
            :alg="form.algorithm"
            :objectives="form.objectives"
          />
        </div>

        <!-- Footer -->
        <div class="modal-footer">
          <template v-if="showCloseConfirm">
            <div class="confirm-row">
              <span class="confirm-msg">Discard settings and close?</span>
              <div class="footer-actions">
                <button class="btn-secondary" @click="cancelClose">Keep editing</button>
                <button class="btn-danger"    @click="confirmClose">Discard & close</button>
              </div>
            </div>
          </template>
          <template v-else>
            <div class="footer-left">
              <span v-if="submitError" class="submit-error">{{ submitError }}</span>
            </div>
            <div class="footer-actions">
              <button v-if="currentStep > 1" class="btn-secondary" :disabled="submitting" @click="prev">
                ← Back
              </button>
              <button class="btn-secondary" :disabled="submitting" @click="requestClose">Cancel</button>
              <button
                v-if="currentStep < TOTAL_STEPS"
                class="btn-primary"
                :disabled="!canProceed"
                @click="next"
              >
                Next →
              </button>
              <button
                v-else
                class="btn-success"
                :disabled="!canProceed || submitting"
                @click="submit"
              >
                <span v-if="submitting">Creating…</span>
                <span v-else>Create experiment</span>
              </button>
            </div>
          </template>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useSyntheticStore } from '../../../app/stores/syntheticStore'
import { createExperiment } from '../../../api/experiments'
import { exportSyntheticExperiment } from '../../../services/exportSyntheticJson'
import type { ObjectiveItem } from '../../../types/simlab'

import SLStep1Review from './steps/SLStep1Review.vue'
import SLStep2Algorithm from './steps/SLStep2Algorithm.vue'
import type { SLStep2Value } from './steps/SLStep2Algorithm.vue'
import SLStep3Objectives from './steps/SLStep3Objectives.vue'
import SLStep4Confirm from './steps/SLStep4Confirm.vue'

const emit = defineEmits<{
  close: []
  created: [experimentId: string]
}>()

const store = useSyntheticStore()

const TOTAL_STEPS = 4
const STEP_LABELS = [
  'Review the benchmark configuration',
  'Configure optimization strategy and algorithm',
  'Define objective names',
  'Confirm and launch',
]
const STEP_SHORT = ['Benchmark', 'Algorithm', 'Objectives', 'Confirm']

const currentStep = ref(1)
const showValidation = ref(false)
const submitting = ref(false)
const submitError = ref<string | null>(null)
const showCloseConfirm = ref(false)

// Build default objective list from the benchmark's M
function buildDefaultObjectives(M: number): ObjectiveItem[] {
  return Array.from({ length: M }, (_, i) => ({ metric_name: `f${i + 1}`, goal: 'min' as const }))
}

const form = reactive<{ algorithm: SLStep2Value; objectives: ObjectiveItem[] }>({
  algorithm: {
    name: `${store.draft.benchmark.toLowerCase()}-nsga3-m${store.draft.M}-n${store.draft.nVars}`,
    strategy: 'nsga3',
    populationSize: 50,
    numberOfGenerations: 20,
    randomSeed: 42,
    probCx: 0.9,
    probMt: 0.1,
    perGeneProb: 0.05,
    selectionMethod: 'tournament',
    crossoverMethod: 'sbx_with_radial_translate',
    mutationMethod: 'polynomial',
    divisions: 10,
  },
  objectives: buildDefaultObjectives(store.draft.M),
})

const canProceed = computed((): boolean => {
  switch (currentStep.value) {
    case 1: return true
    case 2: return (
      form.algorithm.name.trim().length > 0 &&
      form.algorithm.populationSize >= 2 &&
      form.algorithm.numberOfGenerations >= 1
    )
    case 3: return (
      form.objectives.length > 0 &&
      form.objectives.every(o => o.metric_name.trim().length > 0)
    )
    case 4: return true
    default: return false
  }
})

function goToStep(s: number) {
  if (s < currentStep.value) currentStep.value = s
}
function prev() {
  if (currentStep.value > 1) currentStep.value--
  showValidation.value = false
  submitError.value = null
}
function next() {
  if (!canProceed.value) { showValidation.value = true; return }
  showValidation.value = false
  currentStep.value++
}

async function submit() {
  showValidation.value = true
  if (!canProceed.value) return

  const isNsga3 = ['nsga3', 'nsga3_deap', 'nsga3_pymoo'].includes(form.algorithm.strategy)
  const isEvolutionary = form.algorithm.strategy !== 'random_search'

  const algorithm: Record<string, unknown> = {
    population_size: form.algorithm.populationSize,
    number_of_generations: form.algorithm.numberOfGenerations,
    ...(isNsga3 && { divisions: form.algorithm.divisions }),
    ...(isEvolutionary && {
      prob_cx: form.algorithm.probCx,
      prob_mt: form.algorithm.probMt,
      per_gene_prob: form.algorithm.perGeneProb,
      selection_method: form.algorithm.selectionMethod,
      crossover_method: form.algorithm.crossoverMethod,
      mutation_method: form.algorithm.mutationMethod,
    }),
  }

  submitting.value = true
  submitError.value = null
  try {
    const payload = exportSyntheticExperiment({
      draft: store.draft,
      name: form.algorithm.name,
      strategy: form.algorithm.strategy,
      algorithm,
      objectives: form.objectives,
      randomSeed: form.algorithm.randomSeed,
    })
    const experimentId = await createExperiment(payload)
    emit('created', experimentId)
  } catch (e: unknown) {
    submitError.value = e instanceof Error ? e.message : 'Failed to create experiment.'
  } finally {
    submitting.value = false
  }
}

function requestClose() {
  if (submitting.value) return
  showCloseConfirm.value = true
}
function confirmClose() { emit('close') }
function cancelClose()  { showCloseConfirm.value = false }
</script>

<style scoped>
.backdrop {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(15, 23, 42, 0.5);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.modal {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: 0 20px 60px rgba(0,0,0,0.25);
  width: 100%; max-width: 580px;
  max-height: calc(100vh - 48px);
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* Header */
.modal-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  padding: 20px 24px 16px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.header-title { display: flex; align-items: flex-start; gap: 12px; }
.header-icon { font-size: 22px; line-height: 1; color: #f59e0b; margin-top: 2px; }
.title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 0 0 2px; }
.subtitle { font-size: 12px; color: var(--color-text-muted); margin: 0; }
.close-btn {
  background: none; border: none; color: var(--color-text-muted); font-size: 16px;
  cursor: pointer; padding: 4px; border-radius: var(--radius-sm);
  flex-shrink: 0; margin-top: -2px;
}
.close-btn:hover { color: var(--color-text); background: var(--color-bg); }

/* Step indicator */
.step-indicator {
  display: flex; align-items: center;
  padding: 12px 24px 0;
  position: relative; flex-shrink: 0;
}
.step-line {
  position: absolute; top: 28px; left: 52px; right: 52px; height: 1px;
  background: var(--color-border); z-index: 0;
}
.step-dot {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  flex: 1; cursor: default; position: relative; z-index: 1;
}
.step-dot:not(.step-dot--active) { cursor: pointer; }
.step-dot-num {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700;
  background: var(--color-bg); border: 2px solid var(--color-border);
  color: var(--color-text-muted); transition: all 0.15s;
}
.step-dot--done .step-dot-num {
  background: #f59e0b; border-color: #f59e0b; color: #fff;
}
.step-dot--active .step-dot-num {
  background: var(--color-surface); border-color: #f59e0b; color: #f59e0b;
}
.step-dot-label {
  font-size: 10px; font-weight: 500; color: var(--color-text-muted);
  white-space: nowrap;
}
.step-dot--active .step-dot-label { color: #f59e0b; font-weight: 700; }

/* Body */
.modal-body {
  flex: 1; overflow-y: auto;
  padding: 20px 24px;
}

/* Footer */
.modal-footer {
  padding: 14px 24px;
  border-top: 1px solid var(--color-border);
  display: flex; align-items: center; justify-content: space-between;
  flex-shrink: 0;
}
.footer-left { flex: 1; }
.footer-actions { display: flex; gap: 8px; align-items: center; flex-shrink: 0; }
.confirm-row {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%; gap: 12px;
}
.confirm-msg { font-size: 13px; font-weight: 600; color: #d97706; }
.submit-error { font-size: 12px; color: #ef4444; }

/* Buttons */
.btn-primary, .btn-secondary, .btn-success, .btn-danger {
  padding: 8px 16px; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600; cursor: pointer;
  border: 1px solid transparent; transition: background 0.12s, border-color 0.12s;
}
.btn-primary  { background: #f59e0b; color: #fff; border-color: #f59e0b; }
.btn-primary:hover:not(:disabled) { background: #d97706; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-secondary { background: var(--color-bg); color: var(--color-text); border-color: var(--color-border); }
.btn-secondary:hover:not(:disabled) { border-color: var(--color-text-muted); }
.btn-secondary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-success { background: #10b981; color: #fff; border-color: #10b981; }
.btn-success:hover:not(:disabled) { background: #059669; }
.btn-success:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-danger { background: #ef4444; color: #fff; border-color: #ef4444; }
.btn-danger:hover { background: #dc2626; }
</style>
