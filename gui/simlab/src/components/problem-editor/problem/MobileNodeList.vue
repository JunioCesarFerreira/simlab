<template>
  <div class="mobile-node-list">
    <div class="list-header">
      <span>Mobile Nodes ({{ nodes.length }})</span>
      <button @click="addNode">+ Add</button>
    </div>
    <div
      v-for="node in nodes"
      :key="node.id"
      class="node-item"
      :class="{ active: activeNodeId === node.id }"
      @click="editorStore.setActiveNode(node.id)"
    >
      <span class="node-name">{{ node.name }}</span>
      <span class="seg-count">{{ node.segments.length }} seg</span>
      <button class="remove-btn" @click.stop="problemStore.removeMobileNode(node.id)">✕</button>
    </div>
    <div v-if="nodes.length === 0" class="empty">No mobile nodes yet</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useEditorStore } from '../../../app/stores/editorStore'

const problemStore = useProblemStore()
const editorStore = useEditorStore()
const nodes = computed(() => problemStore.draft.mobileNodes)
const activeNodeId = computed(() => editorStore.activeNodeId)

function addNode() {
  const id = problemStore.addMobileNode()
  editorStore.setActiveNode(id)
}
</script>

<style scoped>
.mobile-node-list { display: flex; flex-direction: column; gap: 4px; }
.list-header { display: flex; justify-content: space-between; align-items: center; font-size: 12px; margin-bottom: 2px; }
.list-header button { font-size: 11px; background: #a6e3a1; color: #1e1e2e; border: none; border-radius: 4px; padding: 2px 8px; cursor: pointer; }
.node-item { display: flex; align-items: center; gap: 6px; padding: 5px 8px; background: #313244; border-radius: 4px; cursor: pointer; font-size: 12px; border: 1px solid transparent; }
.node-item.active { background: #45475a; border-color: #89b4fa; }
.node-name { flex: 1; }
.seg-count { color: #a6adc8; font-size: 11px; }
.remove-btn { background: none; border: none; color: #f38ba8; cursor: pointer; font-size: 12px; padding: 0; }
.empty { font-size: 12px; color: #6c7086; text-align: center; padding: 8px; }
</style>
