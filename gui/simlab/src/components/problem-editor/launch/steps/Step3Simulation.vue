<template>
  <div class="step">
    <div class="field-group">
      <label class="field-label" for="sim-duration">Simulation duration (seconds) <span class="required">*</span></label>
      <input
        id="sim-duration"
        :value="modelValue.duration"
        type="number" min="1" step="1"
        @input="updateDuration(($event.target as HTMLInputElement).value)"
      />
    </div>

    <div class="section-divider">Random seeds</div>

    <p class="hint">
      Each seed runs an independent simulation per generation, improving the statistical robustness of results.
    </p>

    <div class="field-group">
      <label class="field-label">Seeds <span class="required">*</span></label>
      <div class="seeds-row">
        <div class="chips">
          <span v-for="(seed, i) in modelValue.randomSeeds" :key="i" class="chip">
            {{ seed }}
            <button class="chip-remove" @click="removeSeed(i)" title="Remove seed">×</button>
          </span>
          <span v-if="modelValue.randomSeeds.length === 0" class="empty-chips">No seeds added.</span>
        </div>
        <div class="add-row">
          <input
            v-model="newSeed"
            type="number"
            step="1"
            placeholder="e.g. 42"
            @keydown.enter.prevent="addSeed"
          />
          <button class="add-btn" @click="addSeed" :disabled="!canAdd">+ Add</button>
        </div>
      </div>
      <span v-if="showError" class="err">Add at least one seed.</span>
    </div>

    <div class="presets">
      <span class="presets-label">Quick presets:</span>
      <button v-for="preset in PRESETS" :key="preset.label" class="preset-btn" @click="applyPreset(preset.seeds)">
        {{ preset.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

export interface Step3Value {
  duration: number
  randomSeeds: number[]
}

const props = defineProps<{ modelValue: Step3Value; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: Step3Value] }>()

const newSeed = ref<string>('')
const canAdd = computed(() => {
  const v = parseInt(newSeed.value, 10)
  return Number.isInteger(v) && !props.modelValue.randomSeeds.includes(v)
})
const showError = computed(() => props.showValidation && props.modelValue.randomSeeds.length === 0)

const PRESETS = [
  { label: '1 seed (42)', seeds: [42] },
  { label: '3 seeds', seeds: [1, 2, 3] },
  { label: '5 seeds', seeds: [1, 2, 3, 4, 5] },
  { label: 'p2 example (8 seeds)', seeds: [336157, 667370, 35239, 873465, 987654, 123456, 493499, 5343] },
]

function updateDuration(raw: string) {
  emit('update:modelValue', { ...props.modelValue, duration: parseInt(raw, 10) || 1 })
}

function addSeed() {
  const v = parseInt(newSeed.value, 10)
  if (!Number.isInteger(v) || props.modelValue.randomSeeds.includes(v)) return
  emit('update:modelValue', { ...props.modelValue, randomSeeds: [...props.modelValue.randomSeeds, v] })
  newSeed.value = ''
}

function removeSeed(i: number) {
  const seeds = [...props.modelValue.randomSeeds]
  seeds.splice(i, 1)
  emit('update:modelValue', { ...props.modelValue, randomSeeds: seeds })
}

function applyPreset(seeds: number[]) {
  emit('update:modelValue', { ...props.modelValue, randomSeeds: seeds })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.field-group { display: flex; flex-direction: column; gap: 6px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
.seeds-row { display: flex; flex-direction: column; gap: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; min-height: 32px; align-items: flex-start; }
.chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; background: var(--color-primary-light);
  border: 1px solid #bfdbfe; border-radius: 99px;
  font-size: 12px; font-weight: 600; color: var(--color-primary); font-family: monospace;
}
.chip-remove {
  background: none; border: none; color: var(--color-primary); cursor: pointer;
  font-size: 14px; line-height: 1; padding: 0; opacity: 0.6;
}
.chip-remove:hover { opacity: 1; }
.empty-chips { font-size: 12px; color: var(--color-text-muted); align-self: center; }
.add-row { display: flex; gap: 8px; }
.add-row input {
  flex: 1; padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none;
}
.add-row input:focus { border-color: var(--color-primary); }
input {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none; width: 100%;
}
input:focus { border-color: var(--color-primary); }
.add-btn {
  padding: 7px 14px; background: var(--color-primary); color: #fff;
  border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; border: none; cursor: pointer;
}
.add-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.presets { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.presets-label { font-size: 11px; color: var(--color-text-muted); }
.preset-btn {
  padding: 4px 10px; font-size: 11px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  color: var(--color-text); cursor: pointer;
}
.preset-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
.err { font-size: 11px; color: #ef4444; }
</style>
