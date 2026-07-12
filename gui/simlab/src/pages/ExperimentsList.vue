<template>
  <div class="list-page">
    <div class="page-header">
      <h1 class="page-title">Experiments</h1>
      <button class="refresh-btn" :disabled="store.loading" @click="load">
        {{ store.loading ? "Loading…" : "Refresh" }}
      </button>
    </div>

    <div v-if="store.error" class="error-banner">
      Failed to load: {{ store.error }}
    </div>

    <FilterBar
      v-model="store.selectedStatus"
      :counts="store.countByStatus"
    />

    <div v-if="store.loading && store.experiments.length === 0" class="loading">
      Loading experiments…
    </div>

    <div v-else-if="store.filtered.length === 0 && !store.loading" class="empty-state">
      No experiments
      <span v-if="store.selectedStatus"> with status "{{ store.selectedStatus }}"</span>.
    </div>

    <div v-else class="exp-list">
      <ExperimentCard
        v-for="e in store.filtered"
        :key="e.id"
        :experiment="e"
        :deleting="deletingId === e.id"
        @delete="onDelete(e)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useExperimentsStore } from "../app/stores/experimentsStore";
import FilterBar from "../components/experiments/FilterBar.vue";
import ExperimentCard from "../components/experiments/ExperimentCard.vue";
import { confirmDialog } from "../composables/useConfirm";
import { reportRuntimeError } from "../composables/useRuntimeError";
import type { ExperimentInfoDto } from "../types/simlab";

const store = useExperimentsStore();

const deletingId = ref<string | null>(null);

async function load() {
  await store.fetchAll();
}

async function onDelete(exp: ExperimentInfoDto) {
  const ok = await confirmDialog({
    title: `Delete experiment "${exp.name}"?`,
    message:
      "This permanently removes the experiment and all its artifacts " +
      "(generations, individuals, simulations and their files). Shared source code is not affected. " +
      "This cannot be undone.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  deletingId.value = exp.id;
  try {
    await store.remove(exp.id);
  } catch (e) {
    reportRuntimeError(e, "Failed to delete experiment");
  } finally {
    deletingId.value = null;
  }
}

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

.refresh-btn:hover:not(:disabled) {
  background: #dbeafe;
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.error-banner {
  padding: 12px 16px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 13px;
  border: 1px solid #fecaca;
}

.exp-list {
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
</style>
