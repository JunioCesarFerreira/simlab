<template>
  <div class="detail-page">
    <div v-if="loading && !campaign" class="loading">
      Loading campaign…
    </div>
    <div v-else-if="error" class="error-banner">
      Error: {{ error }}
    </div>

    <template v-else-if="campaign">
      <!-- Header -->
      <div class="header">
        <RouterLink to="/campaigns" class="back-link">← Campaigns</RouterLink>
        <div class="header-main">
          <div class="header-left">
            <h1 class="campaign-name">{{ campaign.name }}</h1>
            <div v-if="campaign.description" class="description">
              {{ campaign.description }}
            </div>
            <div class="header-meta">
              <span v-if="campaign.created_time" class="meta-date">
                {{ formatDate(campaign.created_time) }}
              </span>
              <span class="exp-count-pill">
                {{ campaign.experiments.length }} experiment{{ campaign.experiments.length !== 1 ? 's' : '' }}
              </span>
            </div>
          </div>
          <div class="header-actions">
            <button class="secondary-btn" @click="openEditModal">Edit</button>
            <button class="action-btn" @click="openAddExperiments">+ Add Experiments</button>
            <button class="danger-btn" :disabled="deleting" @click="confirmDelete">
              {{ deleting ? "Deleting…" : "Delete" }}
            </button>
          </div>
        </div>
      </div>

      <!-- Stats row -->
      <div class="stat-grid" v-if="campaign.experiments.length > 0">
        <div class="stat-card stat-card--running">
          <div class="stat-value">{{ countByStatus['Running'] ?? 0 }}</div>
          <div class="stat-label">Running</div>
        </div>
        <div class="stat-card stat-card--waiting">
          <div class="stat-value">{{ countByStatus['Waiting'] ?? 0 }}</div>
          <div class="stat-label">Queued</div>
        </div>
        <div class="stat-card stat-card--done">
          <div class="stat-value">{{ countByStatus['Done'] ?? 0 }}</div>
          <div class="stat-label">Finished</div>
        </div>
        <div class="stat-card stat-card--error">
          <div class="stat-value">{{ countByStatus['Error'] ?? 0 }}</div>
          <div class="stat-label">Errored</div>
        </div>
      </div>

      <!-- Experiments list -->
      <div class="section-header">
        <span class="section-title">Experiments</span>
      </div>

      <div v-if="campaign.experiments.length === 0" class="empty-state">
        No experiments in this campaign yet.
        <button class="inline-link" @click="openAddExperiments">Add some →</button>
      </div>

      <div v-else class="exp-list">
        <div
          v-for="e in campaign.experiments"
          :key="e.id"
          class="exp-row"
        >
          <ExperimentCard class="exp-card" :experiment="e" />
          <button
            class="remove-btn"
            title="Remove from campaign"
            :disabled="removingId === e.id"
            @click="removeExperiment(e.id)"
          >
            {{ removingId === e.id ? '…' : '✕' }}
          </button>
        </div>
      </div>
    </template>

    <!-- Edit Campaign Modal -->
    <Teleport to="body">
      <div v-if="showEditModal" class="modal-overlay" @click.self="closeEditModal">
        <div class="modal">
          <div class="modal-header">
            <span class="modal-title">Edit Campaign</span>
            <button class="close-btn" @click="closeEditModal">✕</button>
          </div>

          <div class="modal-body modal-body--form">
            <div v-if="editError" class="form-error">{{ editError }}</div>

            <div class="field">
              <label class="field-label">Name <span class="required">*</span></label>
              <input
                ref="editNameInput"
                v-model="editForm.name"
                class="field-input"
                placeholder="Campaign name"
              />
            </div>

            <div class="field">
              <label class="field-label">Description</label>
              <textarea
                v-model="editForm.description"
                class="field-input field-textarea"
                placeholder="Optional description"
                rows="3"
              />
            </div>

            <div class="field">
              <label class="field-label">Date</label>
              <input
                v-model="editForm.date"
                type="datetime-local"
                class="field-input"
              />
            </div>
          </div>

          <div class="modal-footer">
            <button class="cancel-btn" :disabled="saving" @click="closeEditModal">Cancel</button>
            <button class="save-btn" :disabled="saving || !editForm.name.trim()" @click="saveEdit">
              {{ saving ? "Saving…" : "Save" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Add Experiments Modal -->
    <Teleport to="body">
      <div v-if="showAddModal" class="modal-overlay" @click.self="closeAddModal">
        <div class="modal modal--wide">
          <div class="modal-header">
            <span class="modal-title">Add Experiments to Campaign</span>
            <button class="close-btn" @click="closeAddModal">✕</button>
          </div>

          <div class="modal-search">
            <input
              v-model="searchQuery"
              class="search-input"
              placeholder="Search by name…"
            />
          </div>

          <div class="modal-body modal-body--scroll">
            <div v-if="loadingExperiments" class="modal-loading">Loading experiments…</div>
            <div v-else-if="availableExperiments.length === 0" class="modal-empty">
              All experiments are already in this campaign.
            </div>
            <div v-else-if="filteredAvailable.length === 0" class="modal-empty">
              No experiments match "{{ searchQuery }}".
            </div>
            <label
              v-for="exp in filteredAvailable"
              :key="exp.id"
              class="exp-checkbox-row"
            >
              <input
                type="checkbox"
                :value="exp.id"
                v-model="selectedIds"
                class="checkbox"
              />
              <div class="exp-checkbox-info">
                <span class="exp-checkbox-name">{{ exp.name }}</span>
                <StatusBadge :status="exp.status" />
              </div>
            </label>
          </div>

          <div class="modal-footer">
            <span class="selected-count">{{ selectedIds.length }} selected</span>
            <button class="cancel-btn" @click="closeAddModal">Cancel</button>
            <button
              class="save-btn"
              :disabled="selectedIds.length === 0 || addingExperiments"
              @click="addSelected"
            >
              {{ addingExperiments ? "Adding…" : "Add selected" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted } from "vue";
import { useRouter } from "vue-router";
import {
  getCampaignFull,
  updateCampaign,
  addExperimentToCampaign,
  removeExperimentFromCampaign,
  deleteCampaign,
} from "../api/campaigns";
import { getAllExperiments } from "../api/experiments";
import type { CampaignFullDto, ExperimentInfoDto } from "../types/simlab";
import ExperimentCard from "../components/experiments/ExperimentCard.vue";
import StatusBadge from "../components/common/StatusBadge.vue";

const props = defineProps<{ id: string }>();
const router = useRouter();

const campaign = ref<CampaignFullDto | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);
const deleting = ref(false);
const removingId = ref<string | null>(null);

// Edit modal state
const showEditModal = ref(false);
const editNameInput = ref<HTMLInputElement | null>(null);
const saving = ref(false);
const editError = ref<string | null>(null);
const editForm = ref({ name: "", description: "", date: "" });

// Add experiments modal state
const showAddModal = ref(false);
const allExperiments = ref<ExperimentInfoDto[]>([]);
const loadingExperiments = ref(false);
const selectedIds = ref<string[]>([]);
const addingExperiments = ref(false);
const searchQuery = ref("");

const countByStatus = computed(() => {
  const counts: Record<string, number> = {};
  for (const e of campaign.value?.experiments ?? []) {
    counts[e.status] = (counts[e.status] ?? 0) + 1;
  }
  return counts;
});

const linkedIds = computed(() =>
  new Set(campaign.value?.experiments.map((e) => e.id) ?? []),
);

const availableExperiments = computed(() =>
  allExperiments.value.filter((e) => !linkedIds.value.has(e.id)),
);

const filteredAvailable = computed(() => {
  const q = searchQuery.value.trim().toLowerCase();
  if (!q) return availableExperiments.value;
  return availableExperiments.value.filter((e) =>
    e.name.toLowerCase().includes(q),
  );
});

function isoToDatetimeLocal(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso).toISOString().slice(0, 16);
  } catch {
    return "";
  }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function reload() {
  loading.value = true;
  error.value = null;
  try {
    campaign.value = await getCampaignFull(props.id);
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

// ---- Edit modal ----
function openEditModal() {
  editForm.value = {
    name: campaign.value?.name ?? "",
    description: campaign.value?.description ?? "",
    date: isoToDatetimeLocal(campaign.value?.created_time),
  };
  editError.value = null;
  showEditModal.value = true;
}

function closeEditModal() {
  showEditModal.value = false;
  editError.value = null;
}

async function saveEdit() {
  if (!editForm.value.name.trim()) return;
  saving.value = true;
  editError.value = null;
  try {
    await updateCampaign(props.id, {
      name: editForm.value.name.trim(),
      description: editForm.value.description.trim(),
      created_time: editForm.value.date
        ? new Date(editForm.value.date).toISOString()
        : null,
    });
    // Update local state without full reload
    campaign.value!.name = editForm.value.name.trim();
    campaign.value!.description = editForm.value.description.trim();
    campaign.value!.created_time = editForm.value.date
      ? new Date(editForm.value.date).toISOString()
      : null;
    closeEditModal();
  } catch (e: unknown) {
    editError.value = e instanceof Error ? e.message : "Failed to save.";
  } finally {
    saving.value = false;
  }
}

watch(showEditModal, (val) => {
  if (val) nextTick(() => editNameInput.value?.focus());
});

// ---- Remove / Delete ----
async function removeExperiment(experimentId: string) {
  removingId.value = experimentId;
  try {
    await removeExperimentFromCampaign(props.id, experimentId);
    campaign.value!.experiments = campaign.value!.experiments.filter(
      (e) => e.id !== experimentId,
    );
  } finally {
    removingId.value = null;
  }
}

async function confirmDelete() {
  if (!confirm(`Delete campaign "${campaign.value?.name}"? This will not delete the experiments.`)) return;
  deleting.value = true;
  try {
    await deleteCampaign(props.id);
    router.push("/campaigns");
  } finally {
    deleting.value = false;
  }
}

// ---- Add experiments modal ----
async function openAddExperiments() {
  showAddModal.value = true;
  selectedIds.value = [];
  searchQuery.value = "";
  loadingExperiments.value = true;
  try {
    allExperiments.value = await getAllExperiments();
  } finally {
    loadingExperiments.value = false;
  }
}

function closeAddModal() {
  showAddModal.value = false;
  selectedIds.value = [];
  searchQuery.value = "";
}

async function addSelected() {
  if (selectedIds.value.length === 0) return;
  addingExperiments.value = true;
  try {
    await Promise.all(
      selectedIds.value.map((expId) => addExperimentToCampaign(props.id, expId)),
    );
    closeAddModal();
    await reload();
  } finally {
    addingExperiments.value = false;
  }
}

onMounted(reload);
</script>

<style scoped>
.detail-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.back-link {
  font-size: 13px;
  color: var(--color-text-muted);
  text-decoration: none;
  margin-bottom: 4px;
  display: inline-block;
}

.back-link:hover { color: var(--color-primary); }

.header {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.header-main {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.campaign-name {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.2;
}

.description {
  font-size: 14px;
  color: var(--color-text-muted);
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.meta-date {
  font-size: 12px;
  color: var(--color-text-muted);
}

.exp-count-pill {
  font-size: 11px;
  font-weight: 700;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 2px 10px;
}

.header-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-shrink: 0;
}

.secondary-btn {
  padding: 7px 14px;
  border-radius: var(--radius-md);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 13px;
  font-weight: 600;
  border: 1px solid var(--color-border);
  transition: background 0.15s;
}

.secondary-btn:hover { background: var(--color-bg); }

.action-btn {
  padding: 7px 14px;
  border-radius: var(--radius-md);
  background: var(--color-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  transition: opacity 0.15s;
}

.action-btn:hover { opacity: 0.88; }

.danger-btn {
  padding: 7px 14px;
  border-radius: var(--radius-md);
  background: #fee2e2;
  color: var(--status-error);
  font-size: 13px;
  font-weight: 600;
  border: 1px solid #fecaca;
  transition: background 0.15s;
}

.danger-btn:hover:not(:disabled) { background: #fecaca; }
.danger-btn:disabled { opacity: 0.5; cursor: default; }

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 10px;
}

.stat-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  box-shadow: var(--shadow-sm);
}

.stat-value {
  font-size: 28px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 11px;
  color: var(--color-text-muted);
  font-weight: 500;
}

.stat-card--running .stat-value { color: var(--status-running); }
.stat-card--waiting .stat-value { color: var(--status-waiting); }
.stat-card--done .stat-value { color: var(--status-done); }
.stat-card--error .stat-value { color: var(--status-error); }

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.exp-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.exp-row {
  display: flex;
  align-items: stretch;
  gap: 8px;
}

.exp-card {
  flex: 1;
  min-width: 0;
}

.remove-btn {
  flex-shrink: 0;
  width: 32px;
  border-radius: var(--radius-md);
  background: #fee2e2;
  color: var(--status-error);
  font-size: 13px;
  font-weight: 700;
  border: 1px solid #fecaca;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s;
  align-self: center;
}

.remove-btn:hover:not(:disabled) { background: #fecaca; }
.remove-btn:disabled { opacity: 0.5; cursor: default; }

.inline-link {
  margin-left: 6px;
  color: var(--color-primary);
  font-style: normal;
  font-weight: 600;
  font-size: 13px;
}

.inline-link:hover { text-decoration: underline; }

.loading,
.empty-state {
  text-align: center;
  padding: 48px;
  color: var(--color-text-muted);
  font-style: italic;
}

.error-banner {
  padding: 12px 16px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 13px;
  border: 1px solid #fecaca;
}

/* ---- Modal shared ---- */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
}

.modal {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  width: 100%;
  max-width: 480px;
  display: flex;
  flex-direction: column;
  max-height: 80vh;
  overflow: hidden;
}

.modal--wide { max-width: 560px; }

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.modal-title {
  font-size: 15px;
  font-weight: 700;
}

.close-btn {
  font-size: 14px;
  color: var(--color-text-muted);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.close-btn:hover { background: var(--color-border); color: var(--color-text); }

.modal-search {
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.search-input {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font: inherit;
  font-size: 13px;
  background: var(--color-bg);
  outline: none;
}

.search-input:focus { border-color: var(--color-primary); }

.modal-body {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.modal-body--form { gap: 14px; }

.modal-body--scroll {
  overflow-y: auto;
  flex: 1;
}

.modal-loading,
.modal-empty {
  padding: 24px;
  text-align: center;
  color: var(--color-text-muted);
  font-style: italic;
  font-size: 13px;
}

.form-error {
  padding: 8px 12px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 12px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.field-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.required { color: var(--status-error); }

.field-input {
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font: inherit;
  font-size: 13px;
  background: var(--color-bg);
  color: var(--color-text);
  transition: border-color 0.15s;
  outline: none;
  resize: none;
}

.field-input:focus { border-color: var(--color-primary); }
.field-textarea { resize: vertical; }

.exp-checkbox-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background 0.12s;
}

.exp-checkbox-row:hover { background: var(--color-bg); }

.checkbox {
  width: 15px;
  height: 15px;
  flex-shrink: 0;
  accent-color: var(--color-primary);
  cursor: pointer;
}

.exp-checkbox-info {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  flex: 1;
}

.exp-checkbox-name {
  font-size: 13px;
  font-weight: 500;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.modal-footer {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}

.selected-count {
  font-size: 12px;
  color: var(--color-text-muted);
  flex: 1;
}

.cancel-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-muted);
  border: 1px solid var(--color-border);
}

.cancel-btn:hover:not(:disabled) { background: var(--color-bg); }

.save-btn {
  padding: 7px 20px;
  border-radius: var(--radius-md);
  background: var(--color-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
}

.save-btn:hover:not(:disabled) { opacity: 0.88; }
.save-btn:disabled, .cancel-btn:disabled { opacity: 0.5; cursor: default; }
</style>
