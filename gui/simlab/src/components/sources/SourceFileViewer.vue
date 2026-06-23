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
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  /* Ensures the code background fills the entire body area including
     any space below short files, preventing a colour flash on open. */
  background: var(--hljs-code-bg);
}

.viewer-loading, .viewer-error {
  display: flex; align-items: center; justify-content: center;
  height: 120px;
  font-size: 13px; color: var(--color-text-muted);
  background: var(--color-surface);
}
.viewer-error { color: var(--status-error); }

/*
 * Code area
 *
 * The gutter (line-numbers) and the code (code-pre) share the SAME
 * background — var(--hljs-code-bg) inherited from .code-wrap.
 *
 * Why: CSS gives no reliable way to fill the entire scrollable height
 * of an overflow:auto container with a child element's background.
 * min-height:100% inside such a container resolves to the viewport
 * height, not the scroll height, so below long-file content the
 * container's own background bleeds into the gutter area, showing a
 * different colour. The only robust fix is to unify the backgrounds and
 * use border-right + muted text colour for visual gutter separation —
 * the same pattern used by GitHub, GitLab, and VS Code's web viewer.
 */
.code-wrap {
  display: flex;
  flex: 1;
  overflow: auto;
  background: var(--hljs-code-bg);
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 13px;
  line-height: 1.6;
}

.line-numbers {
  display: flex;
  flex-direction: column;
  padding: 16px 12px 16px 16px;
  text-align: right;
  min-width: 48px;
  flex-shrink: 0;
  color: var(--color-text-muted);
  /* font-size must match .code-pre exactly (13px × 1.6 = 20.8 px/line)
     so line numbers stay aligned throughout the file */
  font-size: 13px;
  line-height: 1.6;
  background: transparent;   /* unified with .code-wrap — no mismatch */
  border-right: 1px solid var(--color-border);
  user-select: none;
}

.code-pre {
  margin: 0;
  flex: 1;
  padding: 16px;
  background: transparent;
  color: var(--color-text);
}

.code-block {
  display: block;
  white-space: pre;
}

/* Highlight.js token overrides — driven by CSS vars, theme-aware */
:deep(.hljs-keyword)      { color: var(--hljs-keyword); font-weight: 600; }
:deep(.hljs-built_in)     { color: var(--hljs-builtin); }
:deep(.hljs-type)         { color: var(--hljs-type); }
:deep(.hljs-literal)      { color: var(--hljs-literal); }
:deep(.hljs-number)       { color: var(--hljs-number); }
:deep(.hljs-operator)     { color: var(--hljs-operator); }
:deep(.hljs-punctuation)  { color: var(--hljs-operator); }
:deep(.hljs-string)       { color: var(--hljs-string); }
:deep(.hljs-comment)      { color: var(--hljs-comment); font-style: italic; }
:deep(.hljs-doctag)       { color: var(--hljs-comment); font-style: italic; }
:deep(.hljs-meta)         { color: var(--hljs-meta); }
:deep(.hljs-meta .hljs-string) { color: var(--hljs-string); }
:deep(.hljs-title)        { color: var(--hljs-title); font-weight: 600; }
:deep(.hljs-title.function_) { color: var(--hljs-title); font-weight: 600; }
:deep(.hljs-params)       { color: var(--hljs-params); }
:deep(.hljs-variable)     { color: var(--hljs-params); }
:deep(.hljs-attr)         { color: var(--hljs-type); }
:deep(.hljs-symbol)       { color: var(--hljs-symbol); }
:deep(.hljs-section)      { color: var(--hljs-section); font-weight: 700; }
</style>
