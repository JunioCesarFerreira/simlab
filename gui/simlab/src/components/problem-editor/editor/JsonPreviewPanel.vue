<template>
  <div class="json-preview">
    <div class="header">
      <span>JSON Preview</span>
      <button @click="copyJson" :disabled="errors.length > 0">{{ copied ? 'Copied!' : 'Copy' }}</button>
      <button @click="downloadJson" :disabled="errors.length > 0">Download</button>
      <button class="close-btn" @click="editorStore.toggleJsonPreview()">✕</button>
    </div>
    <div v-if="errors.length > 0" class="errors">
      <div v-for="(e, i) in errors" :key="i" class="error-item">⚠ {{ e.message }}</div>
    </div>
    <pre v-else class="code">{{ jsonText }}</pre>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useEditorStore } from '../../../app/stores/editorStore'
import { exportProblem } from '../../../services/exportProblemJson'
import { validateProblem } from '../../../services/validators'

const problemStore = useProblemStore()
const editorStore = useEditorStore()
const copied = ref(false)

const errors = computed(() => validateProblem(problemStore.draft))
const jsonText = computed(() => {
  if (errors.value.length > 0) return ''
  try {
    return JSON.stringify(exportProblem(problemStore.draft), null, 2)
  } catch (e) {
    return String(e)
  }
})

function copyJson() {
  navigator.clipboard.writeText(jsonText.value)
  copied.value = true
  setTimeout(() => (copied.value = false), 2000)
}

function downloadJson() {
  const blob = new Blob([jsonText.value], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${problemStore.draft.name || 'problem'}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<style scoped>
.json-preview { display: flex; flex-direction: column; height: 100%; background: #181825; }
.header { display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #1e1e2e; border-bottom: 1px solid #313244; }
.header span { flex: 1; font-size: 13px; font-weight: 600; }
.header button { font-size: 11px; background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 3px 8px; cursor: pointer; }
.header button:disabled { opacity: 0.4; cursor: not-allowed; }
.close-btn { color: #f38ba8 !important; }
.code { flex: 1; overflow: auto; padding: 12px; font-size: 11px; color: #a6e3a1; margin: 0; white-space: pre; font-family: 'Cascadia Code', 'Fira Code', monospace; }
.errors { padding: 12px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
.error-item { color: #fab387; font-size: 12px; padding: 4px 0; border-bottom: 1px solid #313244; }
</style>
