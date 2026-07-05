<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="$emit('close')">
      <div class="panel" role="dialog" aria-modal="true" aria-label="Saved problems">

        <!-- Header -->
        <div class="panel-header">
          <div class="header-title">
            <span class="header-icon">💾</span>
            <h2 class="title">Problems</h2>
          </div>
          <button class="close-btn" @click="$emit('close')" aria-label="Close">✕</button>
        </div>

        <!-- ── SAVE SECTION ─────────────────────────────────────────────────── -->
        <div class="section">
          <div class="section-title">Save current problem</div>

          <div class="field-row">
            <label class="field-label">Name</label>
            <input
              v-model="saveName"
              class="name-input"
              placeholder="e.g. Urban P2 scenario A"
              @keydown.enter="saveProblem"
            />
          </div>

          <div v-if="hasBackground" class="field-row checkbox-row">
            <label class="checkbox-label">
              <input type="checkbox" v-model="includeBackground" />
              Include background image
            </label>
          </div>

          <div class="save-actions">
            <button
              class="btn-primary"
              :disabled="!saveName.trim() || saving"
              @click="saveProblem"
            >
              {{ saving ? 'Saving…' : '💾 Save' }}
            </button>
            <span v-if="saveError" class="msg-error">{{ saveError }}</span>
            <span v-if="saveOk" class="msg-ok">✓ Saved</span>
          </div>
        </div>

        <div class="divider" />

        <!-- ── LOAD SECTION ─────────────────────────────────────────────────── -->
        <div class="section">
          <div class="section-title-row">
            <span class="section-title">Saved problems</span>
            <span v-if="!listLoading" class="count-badge">{{ problems.length }}</span>
            <button class="refresh-btn" @click="loadList" :disabled="listLoading" title="Refresh list">↻</button>
          </div>

          <div v-if="listLoading" class="state-placeholder">Loading…</div>
          <div v-else-if="listError" class="state-placeholder state-error">{{ listError }}</div>
          <div v-else-if="!problems.length" class="state-placeholder">No saved problems yet.</div>

          <div v-else class="problem-list">
            <div
              v-for="p in problems"
              :key="p.id"
              class="problem-item"
              :class="{ 'is-loading': loadingId === p.id }"
            >
              <div class="item-info">
                <span class="item-name">{{ p.name }}</span>
                <span class="item-meta">
                  <span v-if="p.has_background" class="bg-badge" title="Has background image">🖼</span>
                  {{ formatDate(p.updated_time) }}
                </span>
              </div>
              <div class="item-actions">
                <button
                  class="btn-load"
                  :disabled="!!loadingId || !!deletingId"
                  @click="loadProblem(p)"
                >
                  {{ loadingId === p.id ? 'Loading…' : 'Load' }}
                </button>
                <button
                  class="btn-delete"
                  :disabled="!!loadingId || !!deletingId"
                  :title="`Delete '${p.name}'`"
                  @click="confirmDelete(p)"
                >
                  {{ deletingId === p.id ? '…' : '✕' }}
                </button>
              </div>
            </div>
          </div>

          <span v-if="loadError" class="msg-error">{{ loadError }}</span>
        </div>

        <!-- Delete confirmation -->
        <div v-if="pendingDelete" class="confirm-bar">
          <span class="confirm-msg">Delete <strong>{{ pendingDelete.name }}</strong>?</span>
          <div class="confirm-actions">
            <button class="btn-secondary" @click="pendingDelete = null">Cancel</button>
            <button class="btn-danger" @click="doDelete">Delete</button>
          </div>
        </div>

      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useProblemStore } from '../../../app/stores/problemStore';
import { useEditorStore } from '../../../app/stores/editorStore';
import {
  listProblems,
  createProblem,
  getProblem,
  deleteProblem,
  uploadBackground,
  downloadBackground,
  type ProblemInfoDto,
} from '../../../api/problems';

defineEmits<{ close: [] }>();

const problemStore = useProblemStore();
const editorStore = useEditorStore();

// ── Save state ──────────────────────────────────────────────────────────────
const saveName = ref(problemStore.draft.name ?? '');
const includeBackground = ref(true);
const saving = ref(false);
const saveError = ref('');
const saveOk = ref(false);

const hasBackground = computed(() => !!editorStore.backgroundImage);

async function saveProblem() {
  const name = saveName.value.trim();
  if (!name || saving.value) return;

  saving.value = true;
  saveError.value = '';
  saveOk.value = false;

  try {
    const id = await createProblem(
      name,
      problemStore.draft,
      editorStore.imageWorldBounds,
    );

    if (includeBackground.value && editorStore.backgroundImage) {
      try {
        await uploadBackground(id, editorStore.backgroundImage);
      } catch (bgErr) {
        saveError.value = bgErr instanceof Error ? bgErr.message : 'Background upload failed.';
      }
    }

    saveOk.value = true;
    setTimeout(() => { saveOk.value = false; }, 3000);
    await loadList();
  } catch (e) {
    saveError.value = e instanceof Error ? e.message : 'Save failed.';
  } finally {
    saving.value = false;
  }
}

// ── List state ──────────────────────────────────────────────────────────────
const problems = ref<ProblemInfoDto[]>([]);
const listLoading = ref(false);
const listError = ref('');

async function loadList() {
  listLoading.value = true;
  listError.value = '';
  try {
    problems.value = await listProblems();
  } catch (e) {
    listError.value = e instanceof Error ? e.message : 'Failed to load list.';
  } finally {
    listLoading.value = false;
  }
}

// ── Load a problem ──────────────────────────────────────────────────────────
const loadingId = ref<string | null>(null);
const loadError = ref('');

async function loadProblem(info: ProblemInfoDto) {
  if (loadingId.value) return;
  loadingId.value = info.id;
  loadError.value = '';

  try {
    const full = await getProblem(info.id);

    // Load draft into store
    problemStore.loadDraft(full.draft);

    // Restore image world bounds
    editorStore.setImageWorldBounds(
      full.image_world_bounds
        ? (full.image_world_bounds as [number, number, number, number])
        : null,
    );

    // Load background image if present
    if (full.background_image_id) {
      try {
        const dataUrl = await downloadBackground(info.id);
        editorStore.setBackground(dataUrl);
      } catch {
        editorStore.setBackground(null);
      }
    } else {
      editorStore.setBackground(null);
    }
  } catch (e) {
    loadError.value = e instanceof Error ? e.message : 'Failed to load problem.';
  } finally {
    loadingId.value = null;
  }
}

// ── Delete ──────────────────────────────────────────────────────────────────
const pendingDelete = ref<ProblemInfoDto | null>(null);
const deletingId = ref<string | null>(null);

function confirmDelete(p: ProblemInfoDto) {
  pendingDelete.value = p;
}

async function doDelete() {
  if (!pendingDelete.value) return;
  const p = pendingDelete.value;
  pendingDelete.value = null;
  deletingId.value = p.id;
  try {
    await deleteProblem(p.id);
    await loadList();
  } catch {
    // keep list as-is; next refresh will fix
  } finally {
    deletingId.value = null;
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────
function formatDate(iso: string): string {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

onMounted(loadList);
</script>

<style scoped>
/* ── Backdrop & panel ─────────────────────────────────────────────────────── */
.backdrop {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(15, 23, 42, 0.45);
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}

.panel {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
  width: 100%; max-width: 480px;
  max-height: calc(100vh - 48px);
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* ── Header ───────────────────────────────────────────────────────────────── */
.panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.header-title { display: flex; align-items: center; gap: 10px; }
.header-icon { font-size: 20px; line-height: 1; }
.title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 0; }
.close-btn {
  background: none; border: none; font-size: 15px; color: var(--color-text-muted);
  cursor: pointer; padding: 4px; border-radius: 5px; line-height: 1;
}
.close-btn:hover { color: var(--color-text); background: var(--color-bg); }

/* ── Sections ─────────────────────────────────────────────────────────────── */
.section {
  padding: 16px 20px;
  display: flex; flex-direction: column; gap: 10px;
}

.section:last-of-type {
  flex: 1; overflow-y: auto; min-height: 0;
}

.section-title {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--color-text-muted);
}

.section-title-row {
  display: flex; align-items: center; gap: 8px;
}
.section-title-row .section-title { flex: 1; }

.count-badge {
  font-size: 11px; font-weight: 700;
  background: var(--color-primary-light);
  color: var(--color-primary);
  padding: 1px 7px; border-radius: 99px;
}

.refresh-btn {
  background: none; border: 1px solid var(--color-border);
  border-radius: 5px; padding: 2px 6px; cursor: pointer;
  font-size: 13px; color: var(--color-text-muted);
}
.refresh-btn:hover:not(:disabled) { color: var(--color-text); }
.refresh-btn:disabled { opacity: 0.4; cursor: default; }

.divider {
  height: 1px; background: var(--color-border); flex-shrink: 0;
}

/* ── Save form ────────────────────────────────────────────────────────────── */
.field-row { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 11px; font-weight: 600; color: var(--color-text-muted); }
.name-input {
  padding: 7px 10px; border: 1px solid var(--color-border);
  border-radius: 6px; font-size: 13px; color: var(--color-text);
  background: var(--color-surface); outline: none; width: 100%;
}
.name-input:focus { border-color: var(--color-primary); }

.checkbox-row { flex-direction: row; align-items: center; }
.checkbox-label {
  display: flex; align-items: center; gap: 7px;
  font-size: 12px; color: var(--color-text); cursor: pointer;
}

.save-actions { display: flex; align-items: center; gap: 10px; }

/* ── Problem list ─────────────────────────────────────────────────────────── */
.state-placeholder {
  font-size: 13px; color: var(--color-text-muted); padding: 4px 0;
}
.state-error { color: #ef4444; }

.problem-list { display: flex; flex-direction: column; gap: 4px; }

.problem-item {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px;
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: 7px;
  background: var(--color-bg);
  transition: border-color 0.1s;
}
.problem-item:hover { border-color: var(--color-primary); }
.problem-item.is-loading { opacity: 0.6; }

.item-info { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.item-name {
  font-size: 13px; font-weight: 600; color: var(--color-text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.item-meta {
  font-size: 11px; color: var(--color-text-muted);
  display: flex; align-items: center; gap: 4px;
}
.bg-badge { font-size: 12px; }

.item-actions { display: flex; gap: 6px; flex-shrink: 0; }

/* ── Confirm delete bar ───────────────────────────────────────────────────── */
.confirm-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px;
  background: rgba(234, 179, 8, 0.08);
  border-top: 1px solid rgba(234, 179, 8, 0.3);
  gap: 12px; flex-shrink: 0;
}
.confirm-msg { font-size: 13px; color: #92400e; }
.confirm-actions { display: flex; gap: 8px; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.btn-primary {
  padding: 7px 16px; background: var(--color-primary); color: #fff;
  border: none; border-radius: 6px; font-size: 13px; font-weight: 600;
  cursor: pointer; white-space: nowrap;
}
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-primary:not(:disabled):hover { opacity: 0.88; }

.btn-secondary {
  padding: 6px 12px; border: 1px solid var(--color-border); background: none;
  border-radius: 6px; font-size: 12px; color: var(--color-text); cursor: pointer;
}

.btn-load {
  padding: 5px 12px; background: var(--color-primary); color: #fff;
  border: none; border-radius: 5px; font-size: 12px; font-weight: 600;
  cursor: pointer; white-space: nowrap;
}
.btn-load:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-load:not(:disabled):hover { opacity: 0.85; }

.btn-delete {
  padding: 5px 8px; background: none; color: var(--color-text-muted);
  border: 1px solid var(--color-border); border-radius: 5px; font-size: 12px;
  cursor: pointer;
}
.btn-delete:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-delete:not(:disabled):hover { color: #ef4444; border-color: #ef4444; }

.btn-danger {
  padding: 6px 14px; background: #ef4444; color: #fff;
  border: none; border-radius: 6px; font-size: 12px; font-weight: 600;
  cursor: pointer;
}
.btn-danger:hover { background: #dc2626; }

/* ── Messages ─────────────────────────────────────────────────────────────── */
.msg-error { font-size: 12px; color: #ef4444; }
.msg-ok { font-size: 12px; color: #16a34a; font-weight: 600; }
</style>
