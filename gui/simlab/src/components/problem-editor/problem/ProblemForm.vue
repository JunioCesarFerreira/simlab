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
    </template>
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
      <legend>Sink</legend>
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
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useValidation } from '../../../composables/useValidation'
import { PROBLEM_NAMES, hasCandidates } from '../../../types/problem'
import type { Region } from '../../../types/problem'

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
const canEditSink = computed(() => ['problem1', 'problem2', 'problem3'].includes(problemStore.draft.name))
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
label.error > input { border-color: #f38ba8; }
input, select { padding: 3px 4px; border: 1px solid #444; background: #1e1e2e; color: #cdd6f4; border-radius: 4px; font-size: 11px; width: 100%; min-width: 0; }
fieldset { border: 1px solid #444; border-radius: 4px; padding: 6px; display: flex; flex-direction: column; gap: 4px; }
fieldset.error { border-color: #f38ba8; }
legend { font-size: 11px; color: #a6adc8; }
.region-row { display: grid; grid-template-columns: 14px 60px 60px; gap: 4px; align-items: center; }
.axis-label { font-size: 10px; color: #6c7086; text-align: center; }
.err-msg { font-size: 10px; color: #f38ba8; }
.sink-warn { font-size: 11px; color: #fab387; background: #1e1e2e; border: 1px solid #fab38766; border-radius: 4px; padding: 3px 6px; }
.danger { background: #f38ba8; color: #1e1e2e; border: none; padding: 5px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-top: 4px; }
</style>
