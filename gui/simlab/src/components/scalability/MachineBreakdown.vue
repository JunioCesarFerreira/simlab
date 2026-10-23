<template>
  <section class="machines">
    <h3 class="machines__title">Per physical machine</h3>

    <p v-if="heterogeneous" class="machines__warn">
      The machines in this pool differ in speed — the slowest runs one simulation at
      {{ formatRelativeSpeed(slowest) }} of the fastest's throughput. The pool-wide curve above
      mixes that with scaling: the dispatcher fills the fastest machine first, so past its size
      every added unit is a slower one, and the fit reads that as contention or coherency. The
      curves below are each about one machine's CPUs, which is what the USL describes.
    </p>
    <p v-else class="machines__hint">
      Each machine's own curve: a machine holding <em>k</em> of a wave ran <em>k</em> simulations at
      once, so its throughput at <em>k</em> is <em>k</em> / mean held time over its own runs.
    </p>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Machine</th>
            <th scope="col">Hosts</th>
            <th scope="col" title="One simulation's throughput, relative to the fastest machine">
              Speed
            </th>
            <th scope="col">σ</th>
            <th scope="col">κ</th>
            <th scope="col">Peak</th>
            <th scope="col">Limit now</th>
            <th scope="col"><span class="sr-only">Action</span></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="m in machines"
            :key="m.machine"
            :class="{ selected: m.machine === selected?.machine }"
            @click="selectedName = m.machine"
          >
            <td class="name">{{ m.machine }}</td>
            <td class="hosts">{{ m.hosts.join(", ") }}</td>
            <td>
              {{ formatRelativeSpeed(m.relative_speed) }}
              <span class="sub">{{ formatThroughput(m.unit_throughput) }}/min</span>
            </td>
            <td>{{ formatCoefficient(m.fit.sigma) }}</td>
            <td>{{ formatCoefficient(m.fit.kappa) }}</td>
            <td>
              <template v-if="m.recommended_limit !== null">{{ m.recommended_limit }}</template>
              <template v-else>none ≤ {{ m.max_tested }}</template>
            </td>
            <td>{{ limits[m.machine] ?? "—" }}</td>
            <td>
              <button
                v-if="applicable(m) !== null"
                type="button"
                class="apply-btn"
                :disabled="saving === m.machine"
                @click.stop="apply(m)"
              >
                {{ saving === m.machine ? "Saving…" : `Limit to ${applicable(m)}` }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <p v-if="message" :class="['machines__msg', { 'machines__msg--error': failed }]">
      {{ message }}
    </p>

    <template v-if="selected">
      <p class="machines__reading">
        <strong>{{ selected.machine }}</strong> — {{ machineReading(selected) }}
      </p>
      <UslChart :data="machineChartData(selected)" :name="`${name} · ${selected.machine}`" />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import UslChart from "./UslChart.vue";
import { setMachineLimit } from "../../api/infra";
import {
  formatCoefficient,
  formatRelativeSpeed,
  formatThroughput,
  limitToApply,
  machineChartData,
  machineReading,
  slowestRelative,
} from "../../services/usl";
import type { MachineFit } from "../../types/scalability";

/**
 * The benchmark, one physical machine at a time.
 *
 * The recommended limit is applied from here because this is where the
 * evidence for it is; the Infrastructure page has the same setting for a
 * limit chosen by hand.
 */
const props = defineProps<{
  machines: MachineFit[];
  heterogeneous: boolean;
  /** For the exported chart's file name. */
  name: string;
}>();

const selectedName = ref<string | null>(null);
const saving = ref<string | null>(null);
const message = ref("");
const failed = ref(false);
/** Limits as this page last knew them, so an applied one shows at once. */
const limits = ref<Record<string, number | null>>({});

watch(
  () => props.machines,
  (machines) => {
    limits.value = Object.fromEntries(machines.map((m) => [m.machine, m.limit]));
    if (!machines.some((m) => m.machine === selectedName.value)) {
      // The slowest machine first: it is the one a limit is usually about.
      selectedName.value = machines[machines.length - 1]?.machine ?? null;
    }
  },
  { immediate: true },
);

const selected = computed(
  () => props.machines.find((m) => m.machine === selectedName.value) ?? null,
);
const slowest = computed(() => slowestRelative(props.machines));

function applicable(machine: MachineFit): number | null {
  return limitToApply({ ...machine, limit: limits.value[machine.machine] ?? null });
}

async function apply(machine: MachineFit) {
  const value = applicable(machine);
  if (value === null) return;
  saving.value = machine.machine;
  message.value = "";
  failed.value = false;
  try {
    await setMachineLimit(machine.machine, value);
    limits.value = { ...limits.value, [machine.machine]: value };
    message.value =
      `${machine.machine} now runs at most ${value} simulation(s) at once, across ` +
      `${machine.hosts.length} host(s). It applies from the next placement; change it on the ` +
      `Infrastructure page.`;
  } catch (e) {
    failed.value = true;
    message.value = e instanceof Error ? e.message : String(e);
  } finally {
    saving.value = null;
  }
}
</script>

<style scoped>
.machines {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.machines__title {
  font-size: 14px;
  font-weight: 700;
  margin: 6px 0 0;
}

.machines__hint,
.machines__reading {
  margin: 0;
  font-size: 12px;
  color: var(--color-text-muted);
  line-height: 1.5;
}

.machines__reading {
  font-size: 13px;
  color: var(--color-text);
}

.machines__warn {
  margin: 0;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  font-size: 12px;
  line-height: 1.5;
  background: rgba(245, 158, 11, 0.12);
  color: #b45309;
  border: 1px solid rgba(245, 158, 11, 0.3);
}

.machines__msg {
  margin: 0;
  font-size: 12px;
  color: var(--status-done);
}

.machines__msg--error {
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

tbody tr {
  cursor: pointer;
}

tbody tr.selected {
  background: var(--color-primary-light);
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
  margin-left: 4px;
  font-size: 11px;
  color: var(--color-text-muted);
}

.apply-btn {
  padding: 4px 10px;
  border-radius: var(--radius-md);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  background: var(--color-primary);
  color: #fff;
  border: none;
}

.apply-btn:disabled {
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
