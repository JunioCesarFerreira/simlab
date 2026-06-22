<template>
  <div class="import-panel">
    <div class="header">
      <span>Import JSON</span>
      <button class="close-btn" @click="$emit('close')">✕</button>
    </div>
    <div class="body">
      <p class="hint">Cole o conteúdo de um arquivo <code>problem.json</code> abaixo.</p>
      <textarea
        v-model="raw"
        placeholder='{ "problem": { ... } }'
        spellcheck="false"
      />
      <div v-if="error" class="error">⚠ {{ error }}</div>
      <div v-if="success" class="success">✓ Problema carregado com sucesso</div>
      <div class="actions">
        <button @click="loadFromFile">Abrir arquivo…</button>
        <button class="primary" @click="doImport" :disabled="!raw.trim()">Importar</button>
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
  background: #1e1e2e;
  display: flex; flex-direction: column;
  z-index: 10;
}
.header {
  display: flex; align-items: center; padding: 10px 12px;
  border-bottom: 1px solid #313244; flex-shrink: 0;
}
.header span { flex: 1; font-size: 13px; font-weight: 600; color: #cdd6f4; }
.close-btn { background: none; border: none; color: #f38ba8; cursor: pointer; font-size: 14px; }
.body { flex: 1; display: flex; flex-direction: column; gap: 8px; padding: 12px; overflow: hidden; }
.hint { font-size: 11px; color: #6c7086; }
code { color: #a6adc8; }
textarea {
  flex: 1; resize: none; background: #181825; color: #a6e3a1;
  border: 1px solid #313244; border-radius: 4px;
  padding: 8px; font-size: 11px; font-family: 'Cascadia Code', monospace;
}
.error { font-size: 11px; color: #f38ba8; background: #1e1e2e; border: 1px solid #f38ba866; border-radius: 4px; padding: 4px 8px; }
.success { font-size: 11px; color: #a6e3a1; }
.actions { display: flex; gap: 8px; flex-shrink: 0; }
.actions button { flex: 1; padding: 5px; font-size: 12px; background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; cursor: pointer; }
.actions button.primary { background: #89b4fa; color: #1e1e2e; border-color: #89b4fa; font-weight: 600; }
.actions button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
