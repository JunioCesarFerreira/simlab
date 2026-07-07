<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Saved benchmark configurations</span>
        <button class="close-btn" @click="emit('close')">✕</button>
      </div>

      <div v-if="configs.length === 0" class="empty">
        No saved configurations yet. Use "Save current" to store a configuration.
      </div>

      <div v-else class="list">
        <div v-for="(cfg, i) in configs" :key="i" class="item">
          <div class="item-info">
            <span class="item-name">{{ cfg.name }}</span>
            <span class="item-meta">{{ cfg.benchmark }} · M={{ cfg.M }} · n={{ cfg.nVars }}</span>
          </div>
          <div class="item-actions">
            <button class="btn-load" title="Load" @click="load(cfg)">Load</button>
            <button class="btn-del" title="Delete" @click="remove(i)">✕</button>
          </div>
        </div>
      </div>

      <div class="panel-footer">
        <input
          v-model="saveName"
          placeholder="Configuration name…"
          class="save-input"
          @keydown.enter="save"
        />
        <button class="btn-save" :disabled="!saveName.trim()" @click="save">
          Save current
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useSyntheticStore, type BenchmarkDraft } from '../../app/stores/syntheticStore'

const emit = defineEmits<{ close: [] }>()
const store = useSyntheticStore()

const STORAGE_KEY = 'simlab:synthetic-saved'

interface SavedConfig extends BenchmarkDraft { name: string }

function loadAll(): SavedConfig[] {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') } catch { return [] }
}
function saveAll(list: SavedConfig[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
}

const configs = ref<SavedConfig[]>(loadAll())
const saveName = ref('')

function save() {
  if (!saveName.value.trim()) return
  const entry: SavedConfig = { ...store.draft, name: saveName.value.trim() }
  const list = loadAll()
  list.unshift(entry)
  saveAll(list)
  configs.value = list
  saveName.value = ''
}

function load(cfg: SavedConfig) {
  const { name: _, ...draft } = cfg
  store.setDraft(draft)
  emit('close')
}

function remove(i: number) {
  const list = loadAll()
  list.splice(i, 1)
  saveAll(list)
  configs.value = list
}
</script>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  z-index: 800;
  background: rgba(15, 23, 42, 0.35);
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  padding: 60px 16px 16px;
}
.panel {
  width: 360px;
  max-height: calc(100vh - 80px);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.panel-title { font-size: 13px; font-weight: 700; color: var(--color-text); }
.close-btn {
  background: none; border: none; color: var(--color-text-muted);
  font-size: 14px; cursor: pointer; padding: 2px;
}
.close-btn:hover { color: var(--color-text); }

.empty {
  padding: 20px 16px;
  font-size: 12px;
  color: var(--color-text-muted);
  text-align: center;
}
.list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1px; }
.item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--color-border);
  gap: 8px;
}
.item-info { display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 0; }
.item-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.item-meta { font-size: 11px; color: var(--color-text-muted); font-family: monospace; }
.item-actions { display: flex; gap: 6px; flex-shrink: 0; }
.btn-load {
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm);
  cursor: pointer;
}
.btn-del {
  padding: 4px 8px;
  font-size: 12px;
  background: none;
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.panel-footer {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}
.save-input {
  flex: 1;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-text);
  background: var(--color-surface);
  outline: none;
}
.save-input:focus { border-color: #f59e0b; }
.btn-save {
  padding: 7px 14px;
  font-size: 12px;
  font-weight: 600;
  background: #f59e0b;
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  white-space: nowrap;
}
.btn-save:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
