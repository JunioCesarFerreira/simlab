<template>
  <div class="list-page">
    <div class="page-header">
      <h1 class="page-title">Campaigns</h1>
      <div class="header-actions">
        <button class="refresh-btn" :disabled="store.loading" @click="load">
          {{ store.loading ? "Loading…" : "Refresh" }}
        </button>
        <button class="primary-btn" @click="showModal = true">+ New Campaign</button>
      </div>
    </div>

    <div v-if="store.error" class="error-banner">
      Failed to load: {{ store.error }}
    </div>

    <div v-if="store.loading && store.campaigns.length === 0" class="loading">
      Loading campaigns…
    </div>

    <div v-else-if="store.campaigns.length === 0 && !store.loading" class="empty-state">
      No campaigns yet. Create one to group your experiments.
    </div>

    <div v-else class="campaign-list">
      <CampaignCard
        v-for="c in store.campaigns"
        :key="c.id"
        :campaign="c"
      />
    </div>

    <!-- New Campaign Modal -->
    <Teleport to="body">
      <div v-if="showModal" class="modal-overlay" @click.self="closeModal">
        <div class="modal">
          <div class="modal-header">
            <span class="modal-title">New Campaign</span>
            <button class="close-btn" @click="closeModal">✕</button>
          </div>

          <div class="modal-body">
            <div v-if="formError" class="form-error">{{ formError }}</div>

            <div class="field">
              <label class="field-label">Name <span class="required">*</span></label>
              <input
                ref="nameInput"
                v-model="form.name"
                class="field-input"
                placeholder="Campaign name"
                @keydown.enter="submit"
              />
            </div>

            <div class="field">
              <label class="field-label">Description</label>
              <textarea
                v-model="form.description"
                class="field-input field-textarea"
                placeholder="Optional description"
                rows="3"
              />
            </div>

            <div class="field">
              <label class="field-label">Date</label>
              <input
                v-model="form.date"
                type="datetime-local"
                class="field-input"
              />
            </div>
          </div>

          <div class="modal-footer">
            <button class="cancel-btn" :disabled="saving" @click="closeModal">Cancel</button>
            <button class="save-btn" :disabled="saving || !form.name.trim()" @click="submit">
              {{ saving ? "Creating…" : "Create" }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, onMounted } from "vue";
import { useRouter } from "vue-router";
import { useCampaignsStore } from "../app/stores/campaignsStore";
import { createCampaign } from "../api/campaigns";
import CampaignCard from "../components/campaigns/CampaignCard.vue";

const store = useCampaignsStore();
const router = useRouter();

const showModal = ref(false);
const saving = ref(false);
const formError = ref<string | null>(null);
const nameInput = ref<HTMLInputElement | null>(null);

const emptyForm = () => ({ name: "", description: "", date: "" });
const form = ref(emptyForm());

async function load() {
  await store.fetchAll();
}

function closeModal() {
  showModal.value = false;
  form.value = emptyForm();
  formError.value = null;
}

async function submit() {
  if (!form.value.name.trim()) return;
  saving.value = true;
  formError.value = null;
  try {
    const id = await createCampaign({
      name: form.value.name.trim(),
      description: form.value.description.trim(),
      created_time: form.value.date ? new Date(form.value.date).toISOString() : null,
      experiment_ids: [],
    });
    closeModal();
    await store.fetchAll();
    router.push(`/campaigns/${id}`);
  } catch (e: unknown) {
    formError.value = e instanceof Error ? e.message : "Failed to create campaign.";
  } finally {
    saving.value = false;
  }
}

watch(showModal, (val) => {
  if (val) nextTick(() => nameInput.value?.focus());
});

onMounted(load);
</script>

<style scoped>
.list-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 22px;
  font-weight: 700;
}

.header-actions {
  display: flex;
  gap: 8px;
}

.refresh-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  background: var(--color-primary-light);
  color: var(--color-primary);
  font-size: 13px;
  font-weight: 600;
  border: 1px solid #bfdbfe;
  transition: background 0.15s;
}

.refresh-btn:hover:not(:disabled) { background: #dbeafe; }
.refresh-btn:disabled { opacity: 0.5; cursor: default; }

.primary-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  background: var(--color-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  border: none;
  transition: opacity 0.15s;
}

.primary-btn:hover { opacity: 0.88; }

.error-banner {
  padding: 12px 16px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 13px;
  border: 1px solid #fecaca;
}

.campaign-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.loading,
.empty-state {
  text-align: center;
  padding: 48px;
  color: var(--color-text-muted);
  font-style: italic;
}

/* ---- Modal ---- */
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
  max-width: 440px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
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

.modal-body {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
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

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid var(--color-border);
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
