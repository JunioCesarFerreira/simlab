<template>
  <Teleport to="body">
    <div class="backdrop">
      <div class="modal" role="dialog" aria-modal="true" :aria-label="`Launch experiment — step ${currentStep} of ${TOTAL_STEPS}`">

        <!-- Header -->
        <div class="modal-header">
          <div class="header-title">
            <span class="header-icon">🚀</span>
            <div>
              <h2 class="title">Launch experiment</h2>
              <p class="subtitle">{{ STEP_LABELS[currentStep - 1] }}</p>
            </div>
          </div>
          <button class="close-btn" @click="requestClose" aria-label="Close">✕</button>
        </div>

        <!-- Step indicator -->
        <div class="step-indicator" aria-hidden="true">
          <div
            v-for="s in TOTAL_STEPS"
            :key="s"
            class="step-dot"
            :class="{
              'step-dot--done': s < currentStep,
              'step-dot--active': s === currentStep,
            }"
            @click="goToStep(s)"
          >
            <span class="step-dot-num">{{ s < currentStep ? '✓' : s }}</span>
            <span class="step-dot-label">{{ STEP_SHORT[s - 1] }}</span>
          </div>
          <div class="step-line" />
        </div>

        <!-- Body -->
        <div class="modal-body">
          <Step1Problem v-if="currentStep === 1" />
          <Step2Experiment
            v-else-if="currentStep === 2"
            v-model="form.experiment"
            :repositories="repositories"
            :repositories-loading="repositoriesLoading"
            :show-validation="showValidation"
          />
          <Step3Simulation v-else-if="currentStep === 3" v-model="form.simulation" :show-validation="showValidation" />
          <Step4Objectives v-else-if="currentStep === 4" v-model="form.objectives" :show-validation="showValidation" />
          <Step5DataConversion v-else-if="currentStep === 5" v-model="form.dataConversion" :show-validation="showValidation" />
        </div>

        <!-- Footer -->
        <div class="modal-footer">
          <!-- Confirm-close banner -->
          <template v-if="showCloseConfirm">
            <div class="confirm-row">
              <span class="confirm-msg">⚠ Discard all settings and close?</span>
              <div class="footer-actions">
                <button class="btn-secondary" @click="cancelClose">Keep editing</button>
                <button class="btn-danger" @click="confirmClose">Discard & close</button>
              </div>
            </div>
          </template>

          <!-- Normal footer -->
          <template v-else>
            <div class="footer-left">
              <span v-if="submitError" class="submit-error">{{ submitError }}</span>
            </div>
            <div class="footer-actions">
              <button v-if="currentStep > 1" class="btn-secondary" @click="prev" :disabled="submitting">
                ← Back
              </button>
              <button class="btn-secondary" @click="requestClose" :disabled="submitting">
                Cancel
              </button>
              <button
                v-if="currentStep < TOTAL_STEPS"
                class="btn-primary"
                @click="next"
                :disabled="!canProceed"
              >
                Next →
              </button>
              <button
                v-else
                class="btn-success"
                @click="submit"
                :disabled="!canProceed || submitting"
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
import { ref, reactive, computed, onMounted } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { exportProblem } from '../../../services/exportProblemJson'
import { createExperiment } from '../../../api/experiments'
import { getAllRepositories } from '../../../api/repositories'
import type { DataConversionConfigDto, JsonObject, ObjectiveItem, SourceRepositoryDto } from '../../../types/simlab'

import Step1Problem from './steps/Step1Problem.vue'
import Step2Experiment from './steps/Step2Experiment.vue'
import type { Step2Value } from './steps/Step2Experiment.vue'
import Step3Simulation from './steps/Step3Simulation.vue'
import type { Step3Value } from './steps/Step3Simulation.vue'
// Step3SyntheticValue is used when building the payload below
import Step4Objectives from './steps/Step4Objectives.vue'
import Step5DataConversion from './steps/Step5DataConversion.vue'

const emit = defineEmits<{
  close: []
  created: [experimentId: string]
}>()

const problemStore = useProblemStore()

const TOTAL_STEPS = 5
const STEP_LABELS = [
  'Review the problem defined in the editor',
  'Configure the experiment, repository and algorithm',
  'Configure simulation parameters',
  'Define optimization objectives',
  'Configure data conversion',
]
const STEP_SHORT = ['Problem', 'Experiment', 'Simulation', 'Objectives', 'Data']

const currentStep = ref(1)
const showValidation = ref(false)
const submitting = ref(false)
const submitError = ref<string | null>(null)
const showCloseConfirm = ref(false)

// Source repositories loaded from API
const repositories = ref<SourceRepositoryDto[]>([])
const repositoriesLoading = ref(false)

onMounted(async () => {
  repositoriesLoading.value = true
  try {
    repositories.value = await getAllRepositories()
  } catch {
    // Non-fatal: user can still type a repo ID manually if select is empty
  } finally {
    repositoriesLoading.value = false
  }
})

// Derive initial MAC protocol from the chromosome draft
const initMacProtocol = problemStore.draft.chromosome?.macProtocol ?? 'csma'

// Defaults from post-nsga3-experiment-p2.json
const form = reactive<{
  experiment: Step2Value
  simulation: Step3Value
  objectives: ObjectiveItem[]
  dataConversion: DataConversionConfigDto
}>({
  experiment: {
    name: 'Performing optimization with NSGA-III on problem P2',
    strategy: 'nsga3',
    sourceOptions: [{ protocol: initMacProtocol, repoId: '' }],
    populationSize: 50,
    numberOfGenerations: 10,
    randomSeed: 42,
    probCx: 0.8,
    probMt: 0.15,
    perGeneProb: 0.1,
    selectionMethod: 'tournament',
    crossoverMethod: 'uniform_mask',
    mutationMethod: 'bitflip',
    divisions: 10,
  },
  simulation: {
    duration: 180,
    randomSeeds: [336157, 667370, 35239, 873465, 987654, 123456, 493499, 5343],
    synthetic: { enabled: false, bench: 'DTLZ2', noiseStd: 0 },
  },
  objectives: [
    { metric_name: 'latency',    goal: 'min' },
    { metric_name: 'energy',     goal: 'min' },
    { metric_name: 'throughput', goal: 'max' },
  ],
  dataConversion: {
    node_col: 'node',
    time_col: 'root_time_now',
    metrics: [
      { name: 'cpu_energy_mj',       kind: 'sum_all',              column: 'cpu_energy_mj' },
      { name: 'lpm_energy_mj',       kind: 'sum_all',              column: 'lpm_energy_mj' },
      { name: 'radio_tx_energy_mj',  kind: 'sum_all',              column: 'radio_tx_energy_mj' },
      { name: 'radio_rx_energy_mj',  kind: 'sum_all',              column: 'radio_rx_energy_mj' },
      { name: 'total_sent',          kind: 'sum_last_minus_first', column: 'total_sent' },
      { name: 'total_received',      kind: 'sum_last_minus_first', column: 'total_received' },
      { name: 'server_sent',         kind: 'sum_last_minus_first', column: 'server_received' },
      { name: 'bytes_tx',            kind: 'sum_last_minus_first', column: 'bytes_tx' },
      { name: 'bytes_rx',            kind: 'sum_last_minus_first', column: 'bytes_rx' },
      { name: 'server_bytes_rx',     kind: 'sum_last_minus_first', column: 'server_bytes_rx' },
      { name: 'r2n_latency',         kind: 'mean',                 column: 'r2n_latency' },
      { name: 'n2r_latency',         kind: 'mean',                 column: 'n2r_latency' },
      { name: 'hops',                kind: 'mean',                 column: 'hops' },
      { name: 'rtt_latency',         kind: 'mean',                 column: 'rtt_latency' },
      { name: 'latency',             kind: 'mean',                 column: 'rtt_latency' },
      { name: 'energy',              kind: 'sum_all',              column: 'total_energy_mj' },
      { name: 'throughput',          kind: 'sum_last_minus_first', column: 'server_received' },
    ],
  },
})

const canProceed = computed((): boolean => {
  switch (currentStep.value) {
    case 1: return !!problemStore.draft.sink
    case 2: return (
      form.experiment.name.trim().length > 0 &&
      form.experiment.populationSize >= 2 &&
      form.experiment.numberOfGenerations >= 1 &&
      form.experiment.sourceOptions.length > 0 &&
      form.experiment.sourceOptions.every(o => o.protocol.trim().length > 0 && o.repoId.trim().length > 0)
    )
    case 3: return form.simulation.synthetic.enabled || (form.simulation.randomSeeds.length > 0 && form.simulation.duration >= 1)
    case 4: return (
      form.objectives.length > 0 &&
      form.objectives.every(o => o.metric_name.trim().length > 0)
    )
    case 5: return (
      form.dataConversion.node_col.trim().length > 0 &&
      form.dataConversion.time_col.trim().length > 0 &&
      form.dataConversion.metrics.every(m => m.name.trim() && m.kind && m.column.trim())
    )
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
  if (!canProceed.value) {
    showValidation.value = true
    return
  }
  showValidation.value = false
  currentStep.value++
}

async function submit() {
  showValidation.value = true
  if (!canProceed.value) return

  let exported: ReturnType<typeof exportProblem>
  try {
    exported = exportProblem(problemStore.draft)
  } catch (e: unknown) {
    submitError.value = e instanceof Error ? e.message : 'Invalid problem — cannot export.'
    return
  }

  // Build source_repository_options from the user's mappings
  const sourceRepoOpts: Record<string, string> = {}
  for (const { protocol, repoId } of form.experiment.sourceOptions) {
    if (protocol && repoId) sourceRepoOpts[protocol] = repoId
  }

  submitting.value = true
  submitError.value = null

  const isNsga3 = ['nsga3', 'nsga3_deap', 'nsga3_pymoo'].includes(form.experiment.strategy)
  const isEvolutionary = form.experiment.strategy !== 'random_search'

  try {
    const experimentId = await createExperiment({
      id: null,
      name: form.experiment.name.trim(),
      status: 'Waiting',
      system_message: null,
      created_time: null,
      start_time: null,
      end_time: null,
      generations: [],
      parameters: {
        strategy: form.experiment.strategy,
        algorithm: {
          population_size:       form.experiment.populationSize,
          number_of_generations: form.experiment.numberOfGenerations,
          random_seed:           form.experiment.randomSeed,
          ...(isNsga3 && { divisions: form.experiment.divisions }),
          ...(isEvolutionary && {
            prob_cx:          form.experiment.probCx,
            prob_mt:          form.experiment.probMt,
            per_gene_prob:    form.experiment.perGeneProb,
            selection_method: form.experiment.selectionMethod,
            crossover_method: form.experiment.crossoverMethod,
            mutation_method:  form.experiment.mutationMethod,
          }),
        },
        simulation: {
          duration:     form.simulation.duration,
          random_seeds: form.simulation.randomSeeds,
          ...(form.simulation.synthetic.enabled && {
            synthetic: {
              enabled:   true,
              bench:     form.simulation.synthetic.bench,
              noise_std: form.simulation.synthetic.noiseStd,
            },
          }),
        },
        problem: exported.problem as JsonObject,
        objectives: form.objectives,
      },
      source_repository_options: sourceRepoOpts,
      data_conversion_config: form.dataConversion,
      pareto_front: null,
    })
    emit('created', experimentId)
  } catch (e: unknown) {
    submitError.value = e instanceof Error ? e.message : 'Failed to create experiment. Please try again.'
  } finally {
    submitting.value = false
  }
}

function requestClose() {
  if (submitting.value) return
  showCloseConfirm.value = true
}

function confirmClose() {
  emit('close')
}

function cancelClose() {
  showCloseConfirm.value = false
}
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
  width: 100%; max-width: 600px;
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
.header-icon { font-size: 22px; line-height: 1; margin-top: 2px; }
.title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 0 0 2px; }
.subtitle { font-size: 12px; color: var(--color-text-muted); margin: 0; }
.close-btn {
  background: none; border: none; color: var(--color-text-muted); font-size: 16px;
  cursor: pointer; padding: 4px; border-radius: var(--radius-sm); line-height: 1;
  flex-shrink: 0; margin-top: -2px;
}
.close-btn:hover { color: var(--color-text); background: var(--color-bg); }

/* Step indicator */
.step-indicator {
  display: flex; align-items: center; gap: 0;
  padding: 12px 24px 0;
  position: relative;
  flex-shrink: 0;
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
  border: 2px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text-muted);
  transition: all 0.15s;
}
.step-dot--active .step-dot-num {
  background: var(--color-primary); border-color: var(--color-primary);
  color: #fff; box-shadow: 0 0 0 3px rgba(59,130,246,0.2);
}
.step-dot--done .step-dot-num {
  background: #22c55e; border-color: #22c55e; color: #fff;
}
.step-dot-label { font-size: 10px; color: var(--color-text-muted); white-space: nowrap; }
.step-dot--active .step-dot-label { color: var(--color-primary); font-weight: 600; }
.step-dot--done .step-dot-label { color: #16a34a; }

/* Body */
.modal-body {
  flex: 1; overflow-y: auto; padding: 20px 24px;
}

/* Footer */
.modal-footer {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px;
  border-top: 1px solid var(--color-border);
  flex-shrink: 0; gap: 12px;
}
.footer-left { flex: 1; min-width: 0; }
.footer-actions { display: flex; gap: 8px; flex-shrink: 0; }
.submit-error { font-size: 12px; color: #ef4444; }

.btn-secondary {
  padding: 8px 16px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); background: none;
  font-size: 13px; font-weight: 500; color: var(--color-text); cursor: pointer;
}
.btn-secondary:hover:not(:disabled) { border-color: var(--color-text-muted); }
.btn-primary {
  padding: 8px 18px; border-radius: var(--radius-sm); border: none;
  background: var(--color-primary); color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
}
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-success {
  padding: 8px 18px; border-radius: var(--radius-sm); border: none;
  background: #16a34a; color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
  min-width: 140px;
}
.btn-success:disabled { opacity: 0.4; cursor: not-allowed; }
button:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-danger {
  padding: 8px 16px; border-radius: var(--radius-sm); border: none;
  background: #ef4444; color: #fff;
  font-size: 13px; font-weight: 600; cursor: pointer;
}
.btn-danger:hover { background: #dc2626; }

.confirm-row {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%; gap: 12px;
}
.confirm-msg {
  font-size: 13px; font-weight: 500; color: #b45309;
}
</style>
