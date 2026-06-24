<template>
  <div class="mac-field">
    <span class="label">MAC protocol</span>
    <div class="options">
      <label>
        <input type="radio" :value="'csma'" :checked="current === 'csma'" @change="set('csma')" />
        CSMA/CA
      </label>
      <label>
        <input type="radio" :value="'tsch'" :checked="current === 'tsch'" @change="set('tsch')" />
        TSCH
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import type { MacProtocol } from '../../../types/problem'

const problemStore = useProblemStore()
const current = computed<MacProtocol>(() => problemStore.draft.chromosome?.macProtocol ?? 'csma')

function set(m: MacProtocol) {
  problemStore.setMacProtocol(m)
}
</script>

<style scoped>
.mac-field { display: flex; flex-direction: column; gap: 3px; }
.label { font-size: 12px; color: var(--color-text-muted); }
.options { display: flex; gap: 10px; font-size: 11px; color: var(--color-text); }
.options label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
input[type="radio"] { accent-color: var(--color-primary); }
</style>
