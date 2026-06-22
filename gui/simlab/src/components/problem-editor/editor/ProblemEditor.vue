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
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #181825;
  overflow: hidden;
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
  border-right: 1px solid #313244;
  overflow: hidden;
}
.canvas-area {
  flex: 1;
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.json-overlay {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 420px;
  border-left: 1px solid #313244;
  z-index: 5;
}
</style>
