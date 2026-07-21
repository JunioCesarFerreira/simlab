<template>
  <div class="candidate-list">
    <div class="list-header">
      <span>Candidates ({{ candidates.length }})</span>
    </div>

    <div v-if="candidates.length === 0" class="empty">
      No candidates yet — use the ● tool on the canvas
    </div>

    <div v-for="c in candidates" :key="c.id" class="candidate-row">
      <span class="idx">#{{ candidates.indexOf(c) + 1 }}</span>
      <input
        type="number"
        :value="c.x"
        title="x"
        @change="update(c.id, 'x', +($event.target as HTMLInputElement).value)"
      />
      <input
        type="number"
        :value="c.y"
        title="y"
        @change="update(c.id, 'y', +($event.target as HTMLInputElement).value)"
      />
      <button title="Remove" @click="problemStore.removeCandidate(c.id)">✕</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'

const problemStore = useProblemStore()
const candidates = computed(() => problemStore.draft.candidates)

function update(id: string, axis: 'x' | 'y', value: number) {
  const c = problemStore.draft.candidates.find(c => c.id === id)
  if (!c || !isFinite(value)) return
  if (axis === 'x') problemStore.moveCandidate(id, value, c.y)
  else problemStore.moveCandidate(id, c.x, value)
}
</script>

<style scoped>
.candidate-list { display: flex; flex-direction: column; gap: 3px; }
.list-header { font-size: 12px; color: var(--color-primary); margin-bottom: 2px; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px 0; }
.candidate-row {
  display: grid;
  grid-template-columns: 20px 1fr 1fr 18px;
  gap: 3px;
  align-items: center;
}
.idx { font-size: 10px; color: #9ca3af; text-align: right; }
input {
  padding: 2px 4px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  border-radius: 4px;
  font-size: 11px;
  width: 100%;
  min-width: 0;
}
button {
  background: none;
  border: none;
  color: #ef4444;
  cursor: pointer;
  font-size: 11px;
  padding: 0;
  line-height: 1;
}
</style>
