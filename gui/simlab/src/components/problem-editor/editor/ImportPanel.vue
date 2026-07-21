<template>
  <div class="import-panel">
    <div class="header">
      <span>Import JSON</span>
      <button class="close-btn" @click="$emit('close')">✕</button>
    </div>
    <div class="body">
      <p class="hint">Paste the contents of a <code>problem.json</code> file below.</p>
      <textarea
        v-model="raw"
        placeholder='{ "problem": { ... } }'
        spellcheck="false"
      />
      <div v-if="error" class="error">⚠ {{ error }}</div>
      <div v-if="success" class="success">✓ Problem loaded successfully</div>
      <div class="actions">
        <button @click="loadFromFile">Open file…</button>
        <button class="primary" :disabled="!raw.trim()" @click="doImport">Import</button>
      </div>
      <input ref="fileInput" type="file" accept=".json,application/json" style="display:none" @change="onFile" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { importProblemJson } from '../../../services/importProblemJson'

const emit = defineEmits<{ close: [] }>()
const problemStore = useProblemStore()

const raw = ref('')
const error = ref<string | null>(null)
const success = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

function doImport() {
  error.value = null
  success.value = false
  const result = importProblemJson(raw.value)
  if (!result.ok) { error.value = result.error; return }
  problemStore.loadDraft(result.draft)
  success.value = true
  setTimeout(() => emit('close'), 800)
}

function loadFromFile() {
  fileInput.value?.click()
}

function onFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = ev => { raw.value = ev.target?.result as string }
  reader.readAsText(file)
}
</script>

<style scoped>
.import-panel {
  position: absolute; inset: 0;
  background: var(--color-surface);
  display: flex; flex-direction: column;
  z-index: 10;
}
.header {
  display: flex; align-items: center; padding: 10px 12px;
  border-bottom: 1px solid var(--color-border); flex-shrink: 0;
}
.header span { flex: 1; font-size: 13px; font-weight: 600; color: var(--color-text); }
.close-btn { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 14px; }
.body { flex: 1; display: flex; flex-direction: column; gap: 8px; padding: 12px; overflow: hidden; }
.hint { font-size: 11px; color: #9ca3af; }
code { color: var(--color-text-muted); }
textarea {
  flex: 1; resize: none; background: #f0f9ff; color: #1e40af;
  border: 1px solid var(--color-border); border-radius: 4px;
  padding: 8px; font-size: 11px; font-family: 'Cascadia Code', monospace;
}
.error { font-size: 11px; color: #ef4444; background: var(--color-surface); border: 1px solid #ef444466; border-radius: 4px; padding: 4px 8px; }
.success { font-size: 11px; color: #10b981; }
.actions { display: flex; gap: 8px; flex-shrink: 0; }
.actions button { flex: 1; padding: 5px; font-size: 12px; background: var(--color-border); color: var(--color-text); border: 1px solid #d1d5db; border-radius: 4px; cursor: pointer; }
.actions button.primary { background: var(--color-primary); color: var(--color-surface); border-color: var(--color-primary); font-weight: 600; }
.actions button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
