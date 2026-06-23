<template>
  <Teleport to="body">
    <div class="viewer-overlay" @click.self="$emit('close')">
      <div class="viewer-modal">
        <div class="viewer-header">
          <div class="viewer-title">
            <span class="file-icon">📄</span>
            <span class="file-name">{{ fileName }}</span>
            <span class="lang-badge">{{ langLabel }}</span>
          </div>
          <div class="viewer-actions">
            <button class="action-btn" title="Copy to clipboard" @click="copy">
              {{ copied ? '✓ Copied' : 'Copy' }}
            </button>
            <button class="close-btn" @click="$emit('close')">✕</button>
          </div>
        </div>

        <div class="viewer-body">
          <div v-if="loading" class="viewer-loading">Loading…</div>
          <div v-else-if="error" class="viewer-error">{{ error }}</div>
          <div v-else class="code-wrap">
            <div class="line-numbers" aria-hidden="true">
              <span v-for="n in lineCount" :key="n">{{ n }}</span>
            </div>
            <!-- eslint-disable-next-line vue/no-v-html -->
            <pre class="code-pre"><code class="code-block" v-html="highlighted" /></pre>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import hljs from 'highlight.js/lib/core'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import makefile from 'highlight.js/lib/languages/makefile'
import { getFileContent } from '../../api/repositories'

hljs.registerLanguage('c', c)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('makefile', makefile)

const props = defineProps<{
  repositoryId: string
  fileId: string
  fileName: string
}>()

const emit = defineEmits<{ (e: 'close'): void }>()

// ── Language detection ────────────────────────────────────────────────────────

function detectLang(name: string): { id: string; label: string } {
  const base = name.split('/').pop() ?? name
  const ext = base.includes('.') ? base.split('.').pop()!.toLowerCase() : ''

  if (base.toLowerCase() === 'makefile' || base.toLowerCase() === 'gnumakefile')
    return { id: 'makefile', label: 'Makefile' }
  if (ext === 'c' || ext === 'h') return { id: 'c', label: ext === 'h' ? 'C Header' : 'C' }
  if (ext === 'cpp' || ext === 'cc' || ext === 'cxx' || ext === 'hpp' || ext === 'hh')
    return { id: 'cpp', label: ext.startsWith('h') ? 'C++ Header' : 'C++' }

  return { id: '', label: 'Plain text' }
}

const lang = computed(() => detectLang(props.fileName))
const langLabel = computed(() => lang.value.label)

// ── Content loading ───────────────────────────────────────────────────────────

const rawContent = ref('')
const loading = ref(true)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    rawContent.value = await getFileContent(props.repositoryId, props.fileId)
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Failed to load file.'
  } finally {
    loading.value = false
  }
}

watch(() => props.fileId, load, { immediate: true })

// ── Highlighting ──────────────────────────────────────────────────────────────

const highlighted = computed(() => {
  if (!rawContent.value) return ''
  const langId = lang.value.id
  if (langId && hljs.getLanguage(langId)) {
    return hljs.highlight(rawContent.value, { language: langId }).value
  }
  // Fallback: escape HTML for plain text
  return rawContent.value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
})

const lineCount = computed(() =>
  rawContent.value ? rawContent.value.split('\n').length : 0
)

// ── Copy ──────────────────────────────────────────────────────────────────────

const copied = ref(false)

async function copy() {
  if (!rawContent.value) return
  await navigator.clipboard.writeText(rawContent.value)
  copied.value = true
  setTimeout(() => { copied.value = false }, 1800)
}

// ── Close on Escape ───────────────────────────────────────────────────────────

function onKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => document.addEventListener('keydown', onKeyDown))
onUnmounted(() => document.removeEventListener('keydown', onKeyDown))
</script>

<style scoped>
/* Overlay */
.viewer-overlay {
  position: fixed; inset: 0;
  background: rgba(15, 23, 42, 0.6);
  z-index: 10000;
  display: flex; align-items: center; justify-content: center;
  padding: 32px;
}

.viewer-modal {
  display: flex; flex-direction: column;
  width: 100%; max-width: 900px;
  max-height: calc(100vh - 64px);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.25);
  overflow: hidden;
}

/* Header */
.viewer-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 18px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg);
  flex-shrink: 0;
}

.viewer-title {
  display: flex; align-items: center; gap: 10px;
  overflow: hidden;
}

.file-icon { font-size: 15px; flex-shrink: 0; }

.file-name {
  font-size: 13px; font-weight: 700; color: var(--color-text);
  font-family: 'SFMono-Regular', Consolas, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.lang-badge {
  flex-shrink: 0;
  font-size: 11px; font-weight: 600;
  padding: 2px 8px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-radius: 99px;
}

.viewer-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }

.action-btn {
  padding: 5px 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: none; font-size: 12px;
  color: var(--color-text); cursor: pointer;
}
.action-btn:hover { background: var(--color-surface); }

.close-btn {
  background: none; border: none;
  font-size: 16px; color: var(--color-text-muted);
  cursor: pointer; padding: 4px 8px;
  border-radius: var(--radius-sm);
}
.close-btn:hover { background: var(--color-surface-hover, #f1f5f9); }

/* Body */
.viewer-body {
  flex: 1; overflow: hidden;
  display: flex; flex-direction: column;
}

.viewer-loading, .viewer-error {
  display: flex; align-items: center; justify-content: center;
  height: 120px;
  font-size: 13px; color: var(--color-text-muted);
}
.viewer-error { color: #ef4444; }

/* Code area */
.code-wrap {
  display: flex;
  flex: 1; overflow: auto;
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 13px; line-height: 1.6;
}

.line-numbers {
  display: flex; flex-direction: column;
  padding: 16px 12px 16px 16px;
  text-align: right;
  min-width: 48px;
  color: #94a3b8;
  font-size: 12px; line-height: 1.6;
  background: #f8fafc;
  border-right: 1px solid var(--color-border);
  user-select: none;
  flex-shrink: 0;
}

.code-pre {
  margin: 0; flex: 1;
  padding: 16px;
  overflow: visible;
  background: #fafbfc;
}

.code-block {
  display: block;
  white-space: pre;
}

/* Highlight.js token overrides — light theme matching the GUI palette */
:deep(.hljs-keyword)      { color: #0369a1; font-weight: 600; }
:deep(.hljs-built_in)     { color: #0369a1; }
:deep(.hljs-type)         { color: #7c3aed; }
:deep(.hljs-literal)      { color: #059669; }
:deep(.hljs-number)       { color: #d97706; }
:deep(.hljs-operator)     { color: #475569; }
:deep(.hljs-punctuation)  { color: #475569; }
:deep(.hljs-string)       { color: #059669; }
:deep(.hljs-comment)      { color: #94a3b8; font-style: italic; }
:deep(.hljs-doctag)       { color: #94a3b8; font-style: italic; }
:deep(.hljs-meta)         { color: #b45309; }          /* preprocessor directives */
:deep(.hljs-meta .hljs-string) { color: #059669; }
:deep(.hljs-title)        { color: #1d4ed8; font-weight: 600; }
:deep(.hljs-title.function_) { color: #1d4ed8; font-weight: 600; }
:deep(.hljs-params)       { color: #334155; }
:deep(.hljs-variable)     { color: #334155; }
:deep(.hljs-attr)         { color: #7c3aed; }
:deep(.hljs-symbol)       { color: #dc2626; }
:deep(.hljs-section)      { color: #1d4ed8; font-weight: 700; }  /* Makefile targets */
</style>
