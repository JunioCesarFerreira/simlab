<template>
  <div class="editor-root">
    <SyntheticToolbar
      @launch="showWizard = true"
      @saved="showSaved = true"
      @reset="store.reset()"
    />

    <div class="editor-body">
      <BenchmarkConfigPanel class="panel-left" />
      <BenchmarkPreviewPanel class="panel-right" />
    </div>

    <SavedBenchmarkPanel v-if="showSaved" @close="showSaved = false" />

    <SyntheticLaunchWizard
      v-if="showWizard"
      @close="showWizard = false"
      @created="onCreated"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSyntheticStore } from '../../app/stores/syntheticStore'

import SyntheticToolbar from './SyntheticToolbar.vue'
import BenchmarkConfigPanel from './BenchmarkConfigPanel.vue'
import BenchmarkPreviewPanel from './BenchmarkPreviewPanel.vue'
import SavedBenchmarkPanel from './SavedBenchmarkPanel.vue'
import SyntheticLaunchWizard from './launch/SyntheticLaunchWizard.vue'

const store = useSyntheticStore()
const router = useRouter()

const showWizard = ref(false)
const showSaved  = ref(false)

function onCreated(experimentId: string) {
  showWizard.value = false
  router.push(`/experiments/${experimentId}`)
}
</script>

<style scoped>
.editor-root {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.editor-body {
  flex: 1;
  display: grid;
  grid-template-columns: 380px 1fr;
  overflow: hidden;
}

.panel-left {
  overflow-y: auto;
  border-right: 1px solid var(--color-border);
}

.panel-right {
  overflow-y: auto;
}
</style>
