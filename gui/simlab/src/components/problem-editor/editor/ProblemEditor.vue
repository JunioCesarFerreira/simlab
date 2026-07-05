<template>
  <div class="problem-editor">
    <Toolbar @import="showImport = true" @launch="showLaunch = true" @problems="showProblems = true" />
    <div class="editor-body">
      <PropertiesPanel class="sidebar" />
      <div class="canvas-area">
        <CanvasView />
        <JsonPreviewPanel v-if="showJson" class="json-overlay" />
      </div>
    </div>
    <SavedProblemsPanel v-if="showProblems" @close="showProblems = false" />
    <ImportPanel v-if="showImport" @close="showImport = false" />
    <LaunchWizard
      v-if="showLaunch"
      @close="showLaunch = false"
      @created="onExperimentCreated"
    />
    <PostLaunchDialog
      v-if="createdExperimentId"
      :experiment-id="createdExperimentId"
      @close="createdExperimentId = null"
    />
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
import SavedProblemsPanel from './SavedProblemsPanel.vue'
import LaunchWizard from '../launch/LaunchWizard.vue'
import PostLaunchDialog from '../launch/PostLaunchDialog.vue'

const editorStore = useEditorStore()
const showImport = ref(false)
const showLaunch = ref(false)
const showProblems = ref(false)
const createdExperimentId = ref<string | null>(null)
const showJson = computed(() => editorStore.showJsonPreview)

function onExperimentCreated(experimentId: string) {
  showLaunch.value = false
  createdExperimentId.value = experimentId
}
</script>

<style scoped>
.problem-editor {
  /* Scoped token overrides: medium cool-gray panels, distinct from the main app's pure white */
  --color-bg: #dce3ed;
  --color-surface: #f3f6fb;
  --color-border: #c8d4e2;
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
  overflow-y: auto;
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
