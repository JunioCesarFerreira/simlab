<template>
  <div class="problem-form">
    <h3>Problem</h3>
    <label>
      Name
      <select v-model="name">
        <option v-for="opt in PROBLEM_NAMES" :key="opt" :value="opt">{{ opt }}</option>
      </select>
    </label>
    <label v-if="!hasCandidates(name)">
      Number of Sensors
      <input v-model.number="numSensors" type="number" min="1" />
    </label>
    <template v-if="name === 'problem3'">
      <label>
        Radius of Cover
        <input v-model.number="radiusOfCover" type="number" min="1" />
      </label>
      <label>
        K-Coverage Required
        <input v-model.number="kRequired" type="number" min="1" />
      </label>
      <label :class="{ error: hasError('g_required') }">
        Minimum Neighbours g
        <input v-model.number="gRequired" type="number" min="1" step="1" />
        <span class="hint">Of each deployed mote, the sink included. 1 is implied by connectivity.</span>
        <span v-if="hasError('g_required')" class="err-msg">{{ errorFor('g_required') }}</span>
      </label>
    </template>
    <fieldset v-if="name === 'problem4'" :class="{ error: hasError('p4') }">
      <legend>Mobile sink and motes</legend>
      <span class="hint">The sink is the base; candidates are the stops, targets (◇) the sensor motes.</span>
      <label>Speed (m/s)<input v-model.number="p4.speed" type="number" min="0" step="any" @change="saveP4" /></label>
      <label>Time step (s)<input v-model.number="p4.timeStep" type="number" min="0" step="any" @change="saveP4" /></label>
      <label>Max. route length (stops)<input v-model.number="p4.maxRouteLength" type="number" min="2" step="1" @change="saveP4" /></label>
      <div class="region-row">
        <span class="axis-label">τ</span>
        <input v-model.number="p4.tauMin" type="number" min="0" step="any" placeholder="min (s)" title="Minimum dwell (s)" @change="saveP4" />
        <input v-model.number="p4.tauMax" type="number" min="0" step="any" placeholder="max (s)" title="Maximum dwell (s)" @change="saveP4" />
      </div>
      <label>Data rate (units/s)<input v-model.number="p4.dataRate" type="number" min="0" step="any" @change="saveP4" /></label>
      <label>Buffer capacity (units)<input v-model.number="p4.bufferCapacity" type="number" min="0" step="any" @change="saveP4" /></label>
      <label>Initial energy<input v-model.number="p4.initialEnergy" type="number" min="0" step="any" @change="saveP4" /></label>
      <span v-if="hasError('p4')" class="err-msg">{{ errorFor('p4') }}</span>
    </fieldset>
    <label v-if="hasCoverageConstraint(name)" :class="{ error: hasError('min_coverage_percentage') }">
      Coverage Level α (%)
      <input v-model.number="minCoveragePercentage" type="number" min="0" max="100" step="1" />
      <span class="hint">Minimum share of the sampled trajectory that must stay connected to the sink.</span>
      <span v-if="hasError('min_coverage_percentage')" class="err-msg">{{ errorFor('min_coverage_percentage') }}</span>
    </label>
    <label :class="{ error: hasError('radius_of_reach') }">
      Radius of Reach
      <input v-model.number="radiusOfReach" type="number" min="1" />
      <span v-if="hasError('radius_of_reach')" class="err-msg">{{ errorFor('radius_of_reach') }}</span>
    </label>
    <label :class="{ error: hasError('radius_of_inter') }">
      Radius of Inter
      <input v-model.number="radiusOfInter" type="number" min="1" />
      <span v-if="hasError('radius_of_inter')" class="err-msg">{{ errorFor('radius_of_inter') }}</span>
    </label>
    <fieldset :class="{ error: hasError('region') }">
      <legend>Region</legend>
      <div class="region-row">
        <span class="axis-label">X</span>
        <input v-model.number="region[0]" type="number" placeholder="xmin" />
        <input v-model.number="region[2]" type="number" placeholder="xmax" />
      </div>
      <div class="region-row">
        <span class="axis-label">Y</span>
        <input v-model.number="region[1]" type="number" placeholder="ymin" />
        <input v-model.number="region[3]" type="number" placeholder="ymax" />
      </div>
      <span v-if="hasError('region')" class="err-msg">{{ errorFor('region') }}</span>
    </fieldset>
    <fieldset v-if="canEditSink" :class="{ error: hasError('sink') }">
      <legend>{{ name === 'problem4' ? 'Sink (base)' : 'Sink' }}</legend>
      <div class="region-row">
        <span class="axis-label">XY</span>
        <input
          type="number"
          :value="sinkX"
          placeholder="x"
          @change="updateSink('x', +($event.target as HTMLInputElement).value)"
        />
        <input
          type="number"
          :value="sinkY"
          placeholder="y"
          @change="updateSink('y', +($event.target as HTMLInputElement).value)"
        />
      </div>
      <span v-if="hasError('sink')" class="err-msg">{{ errorFor('sink') }}</span>
    </fieldset>
    <div v-else-if="hasError('sink')" class="sink-warn">⚠ {{ errorFor('sink') }}</div>
    <button class="danger" @click="problemStore.reset()">Reset Problem</button>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useValidation } from '../../../composables/useValidation'
import { DEFAULT_P4, PROBLEM_NAMES, hasCandidates, hasCoverageConstraint } from '../../../types/problem'
import type { P4Settings, Region } from '../../../types/problem'

const problemStore = useProblemStore()
const { hasError, errorFor } = useValidation()

const name = computed({
  get: () => problemStore.draft.name,
  set: v => problemStore.updateMeta({ name: v }),
})
const numSensors = computed({
  get: () => problemStore.draft.numSensors,
  set: v => problemStore.updateMeta({ numSensors: v }),
})
const radiusOfCover = computed({
  get: () => problemStore.draft.radiusOfCover,
  set: v => problemStore.updateMeta({ radiusOfCover: v }),
})
const kRequired = computed({
  get: () => problemStore.draft.kRequired,
  set: v => problemStore.updateMeta({ kRequired: v }),
})
const gRequired = computed({
  get: () => problemStore.draft.gRequired ?? 1,
  set: v => problemStore.updateMeta({ gRequired: v }),
})
// Edited locally and saved on change, so a half-typed number is not stored.
const p4 = reactive<P4Settings>({ ...DEFAULT_P4, ...problemStore.draft.p4 })
watch(() => problemStore.draft.p4, v => Object.assign(p4, { ...DEFAULT_P4, ...v }))
function saveP4() {
  problemStore.updateP4({ ...p4 })
}
const minCoveragePercentage = computed({
  get: () => problemStore.draft.minCoveragePercentage,
  set: v => problemStore.updateMeta({ minCoveragePercentage: v }),
})
const radiusOfReach = computed({
  get: () => problemStore.draft.radiusOfReach,
  set: v => problemStore.updateMeta({ radiusOfReach: v }),
})
const radiusOfInter = computed({
  get: () => problemStore.draft.radiusOfInter,
  set: v => problemStore.updateMeta({ radiusOfInter: v }),
})
const region = computed({
  get: () => problemStore.draft.region,
  set: v => problemStore.updateMeta({ region: v as Region }),
})
const canEditSink = computed(() => ['problem1', 'problem2', 'problem3', 'problem4'].includes(problemStore.draft.name))
const sinkX = computed(() => problemStore.draft.sink?.x ?? '')
const sinkY = computed(() => problemStore.draft.sink?.y ?? '')

function regionCenter(): { x: number; y: number } {
  const [xmin, ymin, xmax, ymax] = problemStore.draft.region
  return {
    x: Math.round((xmin + xmax) / 2),
    y: Math.round((ymin + ymax) / 2),
  }
}

function updateSink(axis: 'x' | 'y', value: number) {
  if (!isFinite(value)) return
  const current = problemStore.draft.sink ?? regionCenter()
  problemStore.setSink({
    x: axis === 'x' ? value : current.x,
    y: axis === 'y' ? value : current.y,
  })
}
</script>

<style scoped>
.problem-form { display: flex; flex-direction: column; gap: 6px; }
label { display: flex; flex-direction: column; font-size: 12px; gap: 2px; }
label.error > input { border-color: #ef4444; }
input, select { padding: 3px 4px; border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-text); border-radius: 4px; font-size: 11px; width: 100%; min-width: 0; }
fieldset { border: 1px solid var(--color-border); border-radius: 4px; padding: 6px; display: flex; flex-direction: column; gap: 4px; }
fieldset.error { border-color: #ef4444; }
legend { font-size: 11px; color: var(--color-text-muted); }
.region-row { display: grid; grid-template-columns: 14px 60px 60px; gap: 4px; align-items: center; }
.axis-label { font-size: 10px; color: #9ca3af; text-align: center; }
.hint { font-size: 10px; color: var(--color-text-muted); line-height: 1.3; }
.err-msg { font-size: 10px; color: #ef4444; }
.sink-warn { font-size: 11px; color: #f97316; background: var(--color-surface); border: 1px solid #f9731666; border-radius: 4px; padding: 3px 6px; }
.danger { background: #ef4444; color: var(--color-surface); border: none; padding: 5px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-top: 4px; }
</style>
