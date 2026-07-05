<template>
  <div class="toolbar">
    <div class="toolbar-left">
      <span class="brand-icon">⬡</span>
      <span class="toolbar-title">Synthetic Instances</span>
      <span class="toolbar-sub">benchmark-driven experiment builder</span>
    </div>
    <div class="toolbar-actions">
      <button class="btn-tool" @click="emit('saved')" title="Saved configurations">
        <span class="btn-icon">⊞</span> Saved
      </button>
      <button class="btn-tool" @click="emit('reset')" title="Reset to defaults">
        <span class="btn-icon">↺</span> Reset
      </button>
      <button class="btn-launch" @click="emit('launch')">
        <span class="btn-icon">▶</span> Launch experiment
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
const emit = defineEmits<{
  launch: []
  saved: []
  reset: []
}>()
</script>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  gap: 12px;
}
.toolbar-left {
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-icon {
  font-size: 20px;
  color: #f59e0b;
}
.toolbar-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--color-text);
}
.toolbar-sub {
  font-size: 11px;
  color: var(--color-text-muted);
}
.toolbar-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.btn-tool {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text);
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s;
}
.btn-tool:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}
.btn-launch {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  background: #f59e0b;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.12s;
}
.btn-launch:hover { background: #d97706; }
.btn-icon { font-size: 13px; }
</style>
