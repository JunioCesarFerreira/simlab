<template>
  <div class="list-page">
    <div class="page-header">
      <h1 class="page-title">Source Repositories</h1>
      <div class="header-actions">
        <button class="refresh-btn" :disabled="store.loading" @click="store.fetchAll()">
          {{ store.loading ? "Loading…" : "Refresh" }}
        </button>
        <button class="primary-btn" @click="openCreate">+ New Repository</button>
      </div>
    </div>

    <div v-if="store.error" class="error-banner">Failed to load: {{ store.error }}</div>

    <div v-if="store.loading && store.repositories.length === 0" class="loading">
      Loading repositories…
    </div>
    <div v-else-if="store.repositories.length === 0 && !store.loading" class="empty-state">
      No source repositories yet. Create one to hold firmware source files.
    </div>
    <div v-else class="repo-grid">
      <RouterLink
        v-for="r in store.repositories"
        :key="r.id"
        :to="`/sources/${r.id}`"
        class="repo-card"
      >
        <div class="repo-card-header">
          <span class="repo-icon">⊛</span>
          <span class="repo-name">{{ r.name }}</span>
        </div>
        <p v-if="r.description" class="repo-desc">{{ r.description }}</p>
        <div class="repo-meta">
          <span class="file-pill">{{ r.source_files.length }} file{{ r.source_files.length !== 1 ? 's' : '' }}</span>
          <span class="repo-id mono" :title="r.id">{{ r.id.slice(-8) }}</span>
        </div>
      </RouterLink>
    </div>

    <!-- Create Modal -->
    <Teleport to="body">
      <div v-if="showCreate" class="modal-overlay" @click.self="closeCreate">
        <div class="modal">
          <div class="modal-header">
            <span class="modal-title">New Source Repository</span>
            <button class="close-btn" @click="closeCreate">✕</button>
          </div>

          <div class="modal-body">
            <div v-if="createError" class="form-error">{{ createError }}</div>

            <div class="field">
              <label class="field-label">Name <span class="required">*</span></label>
              <input
                ref="nameInputRef"
                v-model="createForm.name"
                class="field-input"
                placeholder="e.g. csma-contiki-firmware"
              />
            </div>

            <div class="field">
              <label class="field-label">Description</label>
              <input
                v-model="createForm.description"
                class="field-input"
                placeholder="Brief description of this source"
              />
            </div>

            <div class="field">
              <label class="field-label">Source files <span class="required">*</span></label>
              <label class="file-drop" :class="{ 'file-drop--has': createForm.files.length > 0 }">
                <input
                  type="file"
                  multiple
                  class="file-input-hidden"
                  @change="onFilesSelected"
                />
                <span v-if="createForm.files.length === 0" class="file-drop-hint">
                  Click to select files (.c, .h, Makefile, …)
                </span>
                <span v-else class="file-drop-hint">
                  {{ createForm.files.length }} file{{ createForm.files.length !== 1 ? 's' : '' }} selected — click to change
                </span>
              </label>
              <div v-if="createForm.files.length > 0" class="file-list">
                <div v-for="(f, i) in createForm.files" :key="i" class="file-row">
                  <span class="file-icon">📄</span>
                  <span class="file-name">{{ f.name }}</span>
                  <span class="file-size">{{ formatBytes(f.size) }}</span>
                  <button class="file-remove" @click.prevent="removeSelectedFile(i)">×</button>
                </div>
              </div>
            </div>
          </div>

          <div class="modal-footer">
            <button class="secondary-btn" @click="closeCreate" :disabled="creating">Cancel</button>
            <button class="primary-btn" @click="submitCreate" :disabled="!canCreate || creating">
              {{ creating ? "Creating…" : "Create Repository" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRepositoriesStore } from '../app/stores/repositoriesStore'
import { createRepository } from '../api/repositories'

const store = useRepositoriesStore()

onMounted(() => store.fetchAll())

// ── Create modal ──────────────────────────────────────────────────────────────

const showCreate = ref(false)
const creating = ref(false)
const createError = ref<string | null>(null)
const nameInputRef = ref<HTMLInputElement | null>(null)

const createForm = ref({ name: '', description: '', files: [] as File[] })

const canCreate = computed(() =>
  createForm.value.name.trim().length > 0 && createForm.value.files.length > 0
)

function openCreate() {
  createForm.value = { name: '', description: '', files: [] }
  createError.value = null
  showCreate.value = true
  nextTick(() => nameInputRef.value?.focus())
}

function closeCreate() {
  if (!creating.value) showCreate.value = false
}

function onFilesSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    const incoming = Array.from(input.files)
    const existing = createForm.value.files.map(f => f.name)
    createForm.value.files = [
      ...createForm.value.files,
      ...incoming.filter(f => !existing.includes(f.name)),
    ]
  }
  input.value = ''
}

function removeSelectedFile(i: number) {
  createForm.value.files.splice(i, 1)
}

async function submitCreate() {
  if (!canCreate.value) return
  creating.value = true
  createError.value = null
  try {
    await createRepository(
      createForm.value.name.trim(),
      createForm.value.description.trim(),
      createForm.value.files,
    )
    await store.fetchAll()
    showCreate.value = false
  } catch (e: unknown) {
    createError.value = e instanceof Error ? e.message : 'Failed to create repository.'
  } finally {
    creating.value = false
  }
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<style scoped>
.list-page { display: flex; flex-direction: column; gap: 20px; padding: 24px; max-width: 1100px; margin: 0 auto; }
.page-header { display: flex; align-items: center; justify-content: space-between; }
.page-title { font-size: 22px; font-weight: 700; color: var(--color-text); margin: 0; }
.header-actions { display: flex; gap: 10px; }
.error-banner { padding: 10px 14px; background: #fef2f2; border: 1px solid #fecaca; border-radius: var(--radius-md); font-size: 13px; color: #ef4444; }
.loading, .empty-state { text-align: center; padding: 48px 0; color: var(--color-text-muted); font-size: 14px; }

.repo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.repo-card {
  display: flex; flex-direction: column; gap: 8px;
  padding: 16px 18px;
  background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg);
  transition: border-color 0.12s, box-shadow 0.12s; text-decoration: none; color: inherit;
}
.repo-card:hover { border-color: var(--color-primary); box-shadow: 0 2px 8px rgba(59,130,246,0.1); }
.repo-card-header { display: flex; align-items: center; gap: 10px; }
.repo-icon { font-size: 18px; color: var(--color-primary); flex-shrink: 0; }
.repo-name { font-size: 15px; font-weight: 700; color: var(--color-text); }
.repo-desc { font-size: 12px; color: var(--color-text-muted); margin: 0; line-height: 1.4; }
.repo-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 4px; }
.file-pill {
  font-size: 11px; font-weight: 600; padding: 2px 8px;
  background: var(--color-primary-light); color: var(--color-primary); border-radius: 99px;
}
.repo-id { font-size: 11px; color: var(--color-text-muted); font-family: monospace; }

/* Buttons */
.refresh-btn { padding: 8px 16px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); background: none; font-size: 13px; color: var(--color-text); cursor: pointer; }
.refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.primary-btn { padding: 8px 18px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; cursor: pointer; }
.primary-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.secondary-btn { padding: 8px 16px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); background: none; font-size: 13px; color: var(--color-text); cursor: pointer; }

/* Modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.45); z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 24px; }
.modal { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); box-shadow: 0 20px 60px rgba(0,0,0,0.2); width: 100%; max-width: 500px; display: flex; flex-direction: column; }
.modal-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--color-border); }
.modal-title { font-size: 15px; font-weight: 700; color: var(--color-text); }
.close-btn { background: none; border: none; font-size: 16px; color: var(--color-text-muted); cursor: pointer; padding: 2px 6px; border-radius: var(--radius-sm); }
.close-btn:hover { background: var(--color-bg); color: var(--color-text); }
.modal-body { padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.modal-footer { display: flex; justify-content: flex-end; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--color-border); }
.form-error { padding: 8px 12px; background: #fef2f2; border: 1px solid #fecaca; border-radius: var(--radius-sm); font-size: 12px; color: #ef4444; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
.field-input { padding: 8px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none; }
.field-input:focus { border-color: var(--color-primary); }
.file-drop {
  display: flex; align-items: center; justify-content: center;
  padding: 18px; border: 1.5px dashed var(--color-border); border-radius: var(--radius-md);
  cursor: pointer; transition: border-color 0.12s;
}
.file-drop:hover, .file-drop--has { border-color: var(--color-primary); background: var(--color-primary-light); }
.file-input-hidden { display: none; }
.file-drop-hint { font-size: 12px; color: var(--color-text-muted); text-align: center; }
.file-list { display: flex; flex-direction: column; gap: 4px; }
.file-row { display: grid; grid-template-columns: 18px 1fr auto auto; gap: 8px; align-items: center; padding: 5px 8px; background: var(--color-bg); border-radius: var(--radius-sm); }
.file-icon { font-size: 13px; }
.file-name { font-size: 12px; color: var(--color-text); font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-size { font-size: 11px; color: var(--color-text-muted); white-space: nowrap; }
.file-remove { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 14px; padding: 0 2px; }
</style>
