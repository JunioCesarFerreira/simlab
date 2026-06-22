<template>
  <div class="problem-editor">
    <Toolbar @import="showImport = true" />
    <div class="editor-body">
      <PropertiesPanel class="sidebar" />
      <div class="canvas-area">
        <CanvasView />
        <JsonPreviewPanel v-if="showJson" class="json-overlay" />
      </div>
    </div>
    <ImportPanel v-if="showImport" @close="showImport = false" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useEditorStore } from '../../../app/stores/editorStore'
import Toolbar from './Toolbar.vue'
import PropertiesPanel from './PropertiesPanel.vue'
import CanvasView from './CanvasView.vue'
import JsonPreviewPanel from './JsonPreviewPanel.vue'
import ImportPanel from './ImportPanel.vue'

const editorStore = useEditorStore()
const showImport = ref(false)
const showJson = computed(() => editorStore.showJsonPreview)
</script>

<style scoped>
.problem-editor {
  /* Scoped token overrides: medium cool-gray panels, distinct from the main app's pure white */
  --color-bg: #dce3ed;
  --color-surface: #edf1f7;
  --color-border: #b8c5d4;
  --color-text: #0f172a;
  --color-text-muted: #4b5875;
  --color-primary-light: #dbeafe;

  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg);
  overflow: hidden;
}

/* Inputs and selects stay white so they stand out from panel surfaces */
.problem-editor :deep(input),
.problem-editor :deep(select) {
  background: #ffffff;
  color: var(--color-text);
  border-color: var(--color-border);
}

.editor-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}
.sidebar {
  width: 280px;
  min-width: 220px;
  max-width: 340px;
  flex-shrink: 0;
  border-right: 1px solid var(--color-border);
  overflow: hidden;
}
.canvas-area {
  flex: 1;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  background: #f0f4f9;
}
.json-overlay {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 420px;
  border-left: 1px solid var(--color-border);
  z-index: 5;
}
</style>
