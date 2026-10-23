<template>
  <section class="machines-card">
    <header class="machines-card__head">
      <h2 class="machines-card__title">Physical machines</h2>
      <p class="machines-card__hint">
        Hosts on one machine — VMs on one hypervisor — compete for its CPUs. A limit caps how many
        simulations the machine runs at once across all of them, whatever the placement policy.
        Hosts are grouped by SSH address unless one is given a machine in its settings; the
        scalability page recommends a limit from a benchmark.
      </p>
    </header>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Machine</th>
            <th scope="col">Hosts</th>
            <th scope="col">Running</th>
            <th scope="col">Limit</th>
            <th scope="col"><span class="sr-only">Action</span></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in machines" :key="m.machine">
            <td class="name">{{ m.machine }}</td>
            <td class="hosts">{{ m.hosts.join(", ") }}</td>
            <td>
              {{ m.active }} / {{ m.max_concurrent ?? m.workers }}
              <span class="sub">{{ m.workers }} worker(s)</span>
            </td>
            <td>
              <input
                :value="draftFor(m)"
                type="number"
                min="1"
                :max="Math.max(1, m.workers)"
                placeholder="none"
                :aria-label="`Limit for ${m.machine}`"
                :disabled="saving === m.machine"
                @input="drafts[m.machine] = ($event.target as HTMLInputElement).value"
              />
            </td>
            <td>
              <button
                v-if="changed(m)"
                type="button"
                class="save-btn"
                :disabled="saving === m.machine || !valid(m)"
                @click="save(m)"
              >
                {{ saving === m.machine ? "Saving…" : "Save" }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-if="error" class="machines-card__error">{{ error }}</p>
  </section>
</template>

<script setup lang="ts">
import { reactive, ref } from "vue";
import { setMachineLimit } from "../../api/infra";
import { parseLimit } from "../../services/machines";
import type { MachineDto } from "../../types/infra";

/**
 * Per-machine concurrency limits.
 *
 * The page polls every few seconds, so what is typed lives in `drafts`
 * until it is saved — a refresh must not replace a number under the cursor.
 */
defineProps<{ machines: MachineDto[] }>();
const emit = defineEmits<{ (e: "saved"): void }>();

const drafts = reactive<Record<string, string>>({});
const saving = ref<string | null>(null);
const error = ref("");

function draftFor(m: MachineDto): string {
  return drafts[m.machine] ?? (m.max_concurrent === null ? "" : String(m.max_concurrent));
}

function changed(m: MachineDto): boolean {
  const draft = drafts[m.machine];
  if (draft === undefined) return false;
  const parsed = parseLimit(draft);
  return parsed.ok && parsed.value !== m.max_concurrent;
}

function valid(m: MachineDto): boolean {
  return parseLimit(draftFor(m)).ok;
}

async function save(m: MachineDto) {
  const parsed = parseLimit(draftFor(m));
  if (!parsed.ok) return;
  saving.value = m.machine;
  error.value = "";
  try {
    await setMachineLimit(m.machine, parsed.value);
    delete drafts[m.machine];
    emit("saved");
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    saving.value = null;
  }
}
</script>

<style scoped>
.machines-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.machines-card__title {
  font-size: 14px;
  font-weight: 700;
  margin: 0;
}

.machines-card__hint {
  margin: 4px 0 0;
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-text-muted);
}

.machines-card__error {
  margin: 0;
  font-size: 12px;
  color: var(--status-error);
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

th {
  font-weight: 600;
  color: var(--color-text-muted);
}

td.name {
  font-weight: 600;
  white-space: normal;
  word-break: break-all;
  min-width: 140px;
}

td.hosts {
  white-space: normal;
  min-width: 120px;
}

.sub {
  margin-left: 6px;
  font-size: 11px;
  color: var(--color-text-muted);
}

input {
  width: 72px;
  padding: 4px 6px;
  font-size: 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
}

.save-btn {
  padding: 4px 10px;
  border-radius: var(--radius-md);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  background: var(--color-primary);
  color: #fff;
  border: none;
}

.save-btn:disabled {
  opacity: 0.6;
  cursor: default;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}
</style>
