<template>
  <div class="detail-page">
    <div v-if="loading && !repo" class="loading">Loading repository…</div>
    <div v-else-if="error" class="error-banner">Error: {{ error }}</div>

    <template v-else-if="repo">
      <!-- Header -->
      <div class="header">
        <RouterLink to="/sources" class="back-link">← Source Repositories</RouterLink>
        <div class="header-main">
          <div class="header-left">
            <h1 class="repo-name">{{ repo.name }}</h1>
            <p v-if="repo.description" class="repo-desc">{{ repo.description }}</p>
            <span class="repo-id mono" :title="repo.id">ID: {{ repo.id }}</span>
          </div>
          <div class="header-actions">
            <button class="secondary-btn" @click="openEdit">Edit</button>
            <button class="secondary-btn" :disabled="downloading" @click="triggerDownload">
              {{ downloading ? "Downloading…" : "Download All" }}
            </button>
            <button class="danger-btn" :disabled="deleting" @click="confirmDelete">
              {{ deleting ? "Deleting…" : "Delete" }}
            </button>
          </div>
        </div>
      </div>

      <!-- Files section -->
      <div class="section-header">
        <span class="section-title">
          Source Files
          <span class="count-pill">{{ repo.source_files.length }}</span>
        </span>
        <button class="add-files-btn" @click="openAddFiles">+ Add Files</button>
      </div>

      <div v-if="repo.source_files.length === 0" class="empty-files">
        No files in this repository.
        <button class="inline-link" @click="openAddFiles">Add files →</button>
      </div>

      <div v-else class="file-table">
        <div class="file-table-head">
          <span>File name</span>
          <span>File ID</span>
          <span></span>
        </div>
        <div
          v-for="f in repo.source_files"
          :key="f.id"
          class="file-row"
          :class="{ 'file-row--removing': removingFileId === f.id }"
        >
          <button
            class="file-name-cell file-name-btn"
            :title="`View ${f.file_name}`"
            @click="openViewer(f.id, f.file_name)"
          >
            <span class="file-icon">📄</span>
            {{ f.file_name }}
          </button>
          <span class="file-id mono">{{ f.id }}</span>
          <div class="file-actions">
            <button
              class="view-file-btn"
              :disabled="removingFileId === f.id"
              @click="openViewer(f.id, f.file_name)"
              title="View file"
            >
              &#128065;
            </button>
            <button
              class="remove-file-btn"
              :disabled="removingFileId === f.id"
              @click="removeFile(f.id)"
              title="Remove file"
            >
              {{ removingFileId === f.id ? '…' : '×' }}
            </button>
          </div>
        </div>
      </div>

      <div v-if="removeFileError" class="inline-error">{{ removeFileError }}</div>
    </template>

    <!-- File viewer -->
    <SourceFileViewer
      v-if="viewerFile"
      :repository-id="id"
      :file-id="viewerFile.id"
      :file-name="viewerFile.name"
      @close="viewerFile = null"
    />

    <!-- Edit metadata modal -->
    <Teleport to="body">
      <div v-if="showEdit" class="modal-overlay" @click.self="closeEdit">
        <div class="modal">
          <div class="modal-header">
            <span class="modal-title">Edit Repository</span>
            <button class="close-btn" @click="closeEdit">✕</button>
          </div>
          <div class="modal-body">
            <div v-if="editError" class="form-error">{{ editError }}</div>
            <div class="field">
              <label class="field-label">Name <span class="required">*</span></label>
              <input v-model="editForm.name" class="field-input" placeholder="Repository name" />
            </div>
            <div class="field">
              <label class="field-label">Description</label>
              <input v-model="editForm.description" class="field-input" placeholder="Description" />
            </div>
          </div>
          <div class="modal-footer">
            <button class="secondary-btn" @click="closeEdit" :disabled="saving">Cancel</button>
            <button class="primary-btn" @click="submitEdit" :disabled="!editForm.name.trim() || saving">
              {{ saving ? "Saving…" : "Save" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Add files modal -->
    <Teleport to="body">
      <div v-if="showAddFiles" class="modal-overlay" @click.self="closeAddFiles">
        <div class="modal">
          <div class="modal-header">
            <span class="modal-title">Add Files</span>
            <button class="close-btn" @click="closeAddFiles">✕</button>
          </div>
          <div class="modal-body">
            <div v-if="addFilesError" class="form-error">{{ addFilesError }}</div>
            <div class="field">
              <label class="field-label">Files <span class="required">*</span></label>
              <label class="file-drop" :class="{ 'file-drop--has': addFilesForm.files.length > 0 }">
                <input type="file" multiple class="file-input-hidden" @change="onAddFilesSelected" />
                <span class="file-drop-hint">
                  <template v-if="addFilesForm.files.length === 0">Click to select files</template>
                  <template v-else>{{ addFilesForm.files.length }} file{{ addFilesForm.files.length !== 1 ? 's' : '' }} selected</template>
                </span>
              </label>
              <div v-if="addFilesForm.files.length > 0" class="file-list">
                <div v-for="(f, i) in addFilesForm.files" :key="i" class="file-row-mini">
                  <span class="file-name">{{ f.name }}</span>
                  <span class="file-size">{{ formatBytes(f.size) }}</span>
                  <button class="file-remove" @click.prevent="addFilesForm.files.splice(i, 1)">×</button>
                </div>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button class="secondary-btn" @click="closeAddFiles" :disabled="addingFiles">Cancel</button>
            <button class="primary-btn" @click="submitAddFiles" :disabled="addFilesForm.files.length === 0 || addingFiles">
              {{ addingFiles ? "Uploading…" : "Upload Files" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Delete confirmation modal -->
    <Teleport to="body">
      <div v-if="showDeleteConfirm" class="modal-overlay" @click.self="showDeleteConfirm = false">
        <div class="modal modal--narrow">
          <div class="modal-header">
            <span class="modal-title">Delete Repository</span>
            <button class="close-btn" @click="showDeleteConfirm = false">✕</button>
          </div>
          <div class="modal-body">
            <p class="confirm-text">
              Are you sure you want to delete <strong>{{ repo?.name }}</strong>?
              This will permanently remove all {{ repo?.source_files.length }} file{{ repo?.source_files.length !== 1 ? 's' : '' }} from GridFS.
            </p>
            <div v-if="deleteError" class="form-error">{{ deleteError }}</div>
          </div>
          <div class="modal-footer">
            <button class="secondary-btn" @click="showDeleteConfirm = false" :disabled="deleting">Cancel</button>
            <button class="danger-btn" @click="executeDelete" :disabled="deleting">
              {{ deleting ? "Deleting…" : "Delete permanently" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  getRepository,
  updateRepository,
  addFilesToRepository,
  removeFileFromRepository,
  deleteRepository,
  downloadRepository,
} from '../api/repositories'
import type { SourceRepositoryDto } from '../types/simlab'
import SourceFileViewer from '../components/sources/SourceFileViewer.vue'

const props = defineProps<{ id: string }>()
const router = useRouter()

const repo = ref<SourceRepositoryDto | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    repo.value = await getRepository(props.id)
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)

// ── Download ──────────────────────────────────────────────────────────────────

const downloading = ref(false)

async function triggerDownload() {
  if (!repo.value) return
  downloading.value = true
  try {
    await downloadRepository(props.id, repo.value.name)
  } finally {
    downloading.value = false
  }
}

// ── Edit metadata ─────────────────────────────────────────────────────────────

const showEdit = ref(false)
const saving = ref(false)
const editError = ref<string | null>(null)
const editForm = ref({ name: '', description: '' })

function openEdit() {
  editForm.value = { name: repo.value?.name ?? '', description: repo.value?.description ?? '' }
  editError.value = null
  showEdit.value = true
}

function closeEdit() {
  if (!saving.value) showEdit.value = false
}

async function submitEdit() {
  if (!editForm.value.name.trim()) return
  saving.value = true
  editError.value = null
  try {
    await updateRepository(props.id, {
      name: editForm.value.name.trim(),
      description: editForm.value.description.trim(),
    })
    await load()
    showEdit.value = false
  } catch (e: unknown) {
    editError.value = e instanceof Error ? e.message : 'Failed to save.'
  } finally {
    saving.value = false
  }
}

// ── Add files ─────────────────────────────────────────────────────────────────

const showAddFiles = ref(false)
const addingFiles = ref(false)
const addFilesError = ref<string | null>(null)
const addFilesForm = ref({ files: [] as File[] })

function openAddFiles() {
  addFilesForm.value = { files: [] }
  addFilesError.value = null
  showAddFiles.value = true
}

function closeAddFiles() {
  if (!addingFiles.value) showAddFiles.value = false
}

function onAddFilesSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    const incoming = Array.from(input.files)
    const existing = addFilesForm.value.files.map(f => f.name)
    addFilesForm.value.files = [
      ...addFilesForm.value.files,
      ...incoming.filter(f => !existing.includes(f.name)),
    ]
  }
  input.value = ''
}

async function submitAddFiles() {
  if (addFilesForm.value.files.length === 0) return
  addingFiles.value = true
  addFilesError.value = null
  try {
    await addFilesToRepository(props.id, addFilesForm.value.files)
    await load()
    showAddFiles.value = false
  } catch (e: unknown) {
    addFilesError.value = e instanceof Error ? e.message : 'Failed to upload.'
  } finally {
    addingFiles.value = false
  }
}

// ── Remove single file ────────────────────────────────────────────────────────

const removingFileId = ref<string | null>(null)
const removeFileError = ref<string | null>(null)

async function removeFile(fileId: string) {
  removingFileId.value = fileId
  removeFileError.value = null
  try {
    await removeFileFromRepository(props.id, fileId)
    await load()
  } catch (e: unknown) {
    removeFileError.value = e instanceof Error ? e.message : 'Failed to remove file.'
  } finally {
    removingFileId.value = null
  }
}

// ── File viewer ───────────────────────────────────────────────────────────────

const viewerFile = ref<{ id: string; name: string } | null>(null)

function openViewer(fileId: string, fileName: string) {
  viewerFile.value = { id: fileId, name: fileName }
}

// ── Delete repository ─────────────────────────────────────────────────────────

const showDeleteConfirm = ref(false)
const deleting = ref(false)
const deleteError = ref<string | null>(null)

function confirmDelete() {
  deleteError.value = null
  showDeleteConfirm.value = true
}

async function executeDelete() {
  deleting.value = true
  deleteError.value = null
  try {
    await deleteRepository(props.id)
    router.push('/sources')
  } catch (e: unknown) {
    deleteError.value = e instanceof Error ? e.message : 'Failed to delete.'
    deleting.value = false
  }
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<style scoped>
.detail-page { display: flex; flex-direction: column; gap: 24px; padding: 24px; max-width: 900px; margin: 0 auto; }
.loading, .error-banner { padding: 40px 0; text-align: center; color: var(--color-text-muted); font-size: 14px; }
.error-banner { color: #ef4444; }

/* Header */
.back-link { font-size: 13px; color: var(--color-text-muted); text-decoration: none; }
.back-link:hover { color: var(--color-primary); }
.header { display: flex; flex-direction: column; gap: 10px; }
.header-main { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
.header-left { display: flex; flex-direction: column; gap: 4px; }
.repo-name { font-size: 22px; font-weight: 800; color: var(--color-text); margin: 0; }
.repo-desc { font-size: 13px; color: var(--color-text-muted); margin: 0; }
.repo-id { font-size: 11px; color: var(--color-text-muted); }
.mono { font-family: 'SFMono-Regular', Consolas, monospace; }
.header-actions { display: flex; gap: 8px; flex-shrink: 0; }

/* Section */
.section-header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--color-border); padding-bottom: 8px; }
.section-title { font-size: 13px; font-weight: 700; color: var(--color-text); display: flex; align-items: center; gap: 8px; }
.count-pill { font-size: 11px; background: var(--color-bg); border: 1px solid var(--color-border); border-radius: 99px; padding: 1px 7px; color: var(--color-text-muted); }
.add-files-btn { padding: 6px 14px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius-sm); font-size: 12px; font-weight: 600; cursor: pointer; }

.empty-files { text-align: center; padding: 32px 0; color: var(--color-text-muted); font-size: 13px; }
.inline-link { background: none; border: none; color: var(--color-primary); cursor: pointer; font-size: 13px; padding: 0; }

/* File table */
.file-table { display: flex; flex-direction: column; border: 1px solid var(--color-border); border-radius: var(--radius-md); overflow: hidden; }
.file-table-head {
  display: grid; grid-template-columns: 1fr 1fr 40px;
  padding: 8px 14px; background: var(--color-bg);
  font-size: 11px; font-weight: 600; color: var(--color-text-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
  border-bottom: 1px solid var(--color-border);
}
.file-row {
  display: grid; grid-template-columns: 1fr 1fr 40px;
  padding: 10px 14px; align-items: center;
  border-bottom: 1px solid var(--color-border);
  transition: background 0.1s;
}
.file-row:last-child { border-bottom: none; }
.file-row:hover { background: var(--color-bg); }
.file-row--removing { opacity: 0.5; }
.file-name-cell { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--color-text); }
.file-name-btn {
  background: none; border: none; padding: 0; text-align: left;
  cursor: pointer; color: var(--color-text);
  display: flex; align-items: center; gap: 8px; font-size: 13px;
}
.file-name-btn:hover { color: var(--color-primary); text-decoration: underline; }
.file-icon { font-size: 14px; }
.file-id { font-size: 11px; color: var(--color-text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-actions { display: flex; justify-content: flex-end; gap: 4px; }
.view-file-btn { width: 26px; height: 26px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); background: none; color: var(--color-text-muted); font-size: 13px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.view-file-btn:hover { border-color: var(--color-primary); color: var(--color-primary); background: var(--color-primary-light); }
.view-file-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.remove-file-btn { width: 26px; height: 26px; border: 1px solid #fecaca; border-radius: var(--radius-sm); background: none; color: #ef4444; font-size: 14px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.remove-file-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.inline-error { font-size: 12px; color: #ef4444; }

/* Buttons */
.secondary-btn { padding: 8px 16px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); background: none; font-size: 13px; color: var(--color-text); cursor: pointer; }
.secondary-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.primary-btn { padding: 8px 18px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; cursor: pointer; }
.primary-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.danger-btn { padding: 8px 16px; background: #ef4444; color: #fff; border: none; border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; cursor: pointer; }
.danger-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* Modals */
.modal-overlay { position: fixed; inset: 0; background: rgba(15,23,42,0.45); z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 24px; }
.modal { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); box-shadow: 0 20px 60px rgba(0,0,0,0.2); width: 100%; max-width: 480px; display: flex; flex-direction: column; }
.modal--narrow { max-width: 380px; }
.modal-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--color-border); }
.modal-title { font-size: 15px; font-weight: 700; color: var(--color-text); }
.close-btn { background: none; border: none; font-size: 16px; color: var(--color-text-muted); cursor: pointer; padding: 2px 6px; border-radius: var(--radius-sm); }
.close-btn:hover { background: var(--color-bg); }
.modal-body { padding: 20px; display: flex; flex-direction: column; gap: 14px; }
.modal-footer { display: flex; justify-content: flex-end; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--color-border); }
.form-error { padding: 8px 12px; background: #fef2f2; border: 1px solid #fecaca; border-radius: var(--radius-sm); font-size: 12px; color: #ef4444; }
.confirm-text { font-size: 13px; color: var(--color-text); line-height: 1.5; }
.field { display: flex; flex-direction: column; gap: 5px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
.field-input { padding: 8px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none; }
.field-input:focus { border-color: var(--color-primary); }
.file-drop { display: flex; align-items: center; justify-content: center; padding: 18px; border: 1.5px dashed var(--color-border); border-radius: var(--radius-md); cursor: pointer; transition: border-color 0.12s; }
.file-drop:hover, .file-drop--has { border-color: var(--color-primary); background: var(--color-primary-light); }
.file-input-hidden { display: none; }
.file-drop-hint { font-size: 12px; color: var(--color-text-muted); }
.file-list { display: flex; flex-direction: column; gap: 4px; }
.file-row-mini { display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center; padding: 4px 8px; background: var(--color-bg); border-radius: var(--radius-sm); }
.file-name { font-size: 12px; color: var(--color-text); font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-size { font-size: 11px; color: var(--color-text-muted); }
.file-remove { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 14px; }
</style>
