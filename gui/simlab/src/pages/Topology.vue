<template>
  <div class="topo">
    <div class="page-header">
      <div>
        <h1 class="page-title">Topology</h1>
        <p class="subtitle">
          How the master-node reaches each host and its Cooja workers, and what runs on them now.
          <span v-if="lastUpdated" class="when">Updated {{ lastUpdated }}.</span>
        </p>
      </div>
      <div class="header-actions">
        <label class="auto">
          <input v-model="autoRefresh" type="checkbox" />
          <span>Live</span>
        </label>
        <button class="refresh-btn" :disabled="loading" @click="load()">
          {{ loading ? "Loading…" : "Refresh" }}
        </button>
      </div>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>
    <p v-if="topology && topology.master.status === 'stale'" class="error-banner">
      The master-node has not checked any host for
      {{ formatElapsed(Math.round(topology.master.age_sec ?? 0)) }} — its health cycle is not
      running, so nothing new is being placed and the statuses below may be out of date.
    </p>
    <p v-else-if="topology && topology.master.dns && !topology.master.dns.ok" class="warn-banner">
      The master-node's DNS resolver is not answering
      <template v-if="topology.master.dns.failing.length">
        for {{ topology.master.dns.failing.join(", ") }}</template
      >. Hosts reached by name are not being measured — their status is kept, not counted as
      failures — and those already pinned are still reached at their last known address.
      <template v-if="topology.master.dns.restart_waiting">
        {{ topology.master.dns.restart_waiting }}.
      </template>
      <template v-else-if="!topology.master.dns.self_restart">
        Restart the master-node container (<code>docker restart master_node</code>) to give it the
        machine's current DNS servers.
      </template>
    </p>

    <template v-if="topology">
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value">{{ stats.hostsReachable }}/{{ stats.hosts }}</div>
          <div class="stat-label">Hosts reachable</div>
        </div>
        <div class="stat-card stat-card--busy">
          <div class="stat-value">{{ stats.running }}</div>
          <div class="stat-label">Workers running</div>
        </div>
        <div class="stat-card stat-card--ok">
          <div class="stat-value">{{ stats.free }}</div>
          <div class="stat-label">Workers free</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ topology.running_experiments.length }}</div>
          <div class="stat-label">Experiments running</div>
        </div>
      </div>

      <section class="card">
        <ul class="legend" aria-label="Status colours">
          <li v-for="tone in TONES" :key="tone">
            <i class="dot" :style="{ background: colors[tone] }" />{{ TONE_LABEL[tone] }}
          </li>
          <li v-if="topology.running_experiments.length">
            <i class="dot dot--ring" />ring = experiment it runs
          </li>
        </ul>
        <TopologyGraph :topology="topology" :show-worker-labels="!narrow" />
        <p v-if="!topology.hosts.length" class="empty-state">
          No host registered. Add one on the
          <RouterLink to="/infrastructure">Infrastructure</RouterLink> page.
        </p>
      </section>

      <section class="card">
        <h2 class="card-title">Running experiments</h2>
        <p v-if="!topology.running_experiments.length" class="empty-state">
          Nothing is running. Workers in green are free to take work.
        </p>
        <ul v-else class="runs">
          <li v-for="exp in topology.running_experiments" :key="exp.id" class="run">
            <i class="swatch" :style="{ borderColor: expColors[exp.id] }" />
            <RouterLink :to="`/experiments/${exp.id}`" class="run__name">{{ exp.name }}</RouterLink>
            <span class="run__strategy">{{ exp.strategy || "" }}</span>
            <span class="run__where">
              <template v-if="exp.on_workers">
                {{ exp.on_workers }} worker(s) ·
                {{
                  Object.entries(exp.by_host)
                    .map(([h, n]) => `${h} ${n}`)
                    .join(" · ")
                }}
              </template>
              <template v-else>waiting for a worker</template>
            </span>
          </li>
        </ul>
      </section>
    </template>
    <p v-else-if="loading" class="empty-state">Loading topology…</p>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import TopologyGraph from "../components/infra/TopologyGraph.vue";
import { getTopology } from "../api/infra";
import { useTheme } from "../composables/useTheme";
import {
  TONE_LABEL,
  experimentColors,
  formatElapsed,
  summarize,
  toneColors,
  type Tone,
} from "../services/topology";
import type { TopologyDto } from "../types/infra";

/** Fast enough to watch a generation move through the pool; a placement or a
 *  release shows up within one tick. */
const POLL_MS = 3000;
const TONES: Tone[] = ["ok", "busy", "warn", "bad", "off"];

const topology = ref<TopologyDto | null>(null);
const loading = ref(false);
const error = ref("");
const lastUpdated = ref("");
const autoRefresh = ref(true);
const narrow = ref(typeof window !== "undefined" && window.innerWidth < 720);
const { isDark } = useTheme();

const stats = computed(() =>
  topology.value
    ? summarize(topology.value)
    : { hosts: 0, hostsReachable: 0, workers: 0, running: 0, free: 0 },
);
const colors = computed(() => toneColors(isDark.value));
const expColors = computed(() =>
  topology.value ? experimentColors(topology.value.running_experiments) : {},
);

let poll: ReturnType<typeof setInterval> | null = null;
let inFlight: AbortController | null = null;

async function load(showSpinner = true) {
  if (showSpinner) loading.value = true;
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  try {
    topology.value = await getTopology(controller.signal);
    lastUpdated.value = new Date().toLocaleTimeString();
    error.value = "";
  } catch (e) {
    if (!controller.signal.aborted) error.value = e instanceof Error ? e.message : String(e);
  } finally {
    if (inFlight === controller) inFlight = null;
    loading.value = false;
  }
}

function onResize() {
  narrow.value = window.innerWidth < 720;
}

onMounted(async () => {
  window.addEventListener("resize", onResize);
  await load();
  poll = setInterval(() => {
    // Not while the tab is hidden: nobody is looking, and the requests would
    // only add load to the API during a run.
    if (autoRefresh.value && !document.hidden) load(false);
  }, POLL_MS);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
  if (poll) clearInterval(poll);
  inFlight?.abort();
});
</script>

<style scoped>
.topo {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.page-title {
  font-size: 22px;
  font-weight: 700;
}

.subtitle {
  font-size: 13px;
  color: var(--color-text-muted);
  margin-top: 2px;
}

.when {
  font-style: italic;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.auto {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--color-text-muted);
  cursor: pointer;
}

.refresh-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  background: var(--color-primary-light);
  color: var(--color-link, var(--color-primary));
  border: 1px solid #bfdbfe;
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.error-banner {
  padding: 12px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.5;
  background: #fee2e2;
  color: var(--status-error);
  border: 1px solid #fecaca;
}

.warn-banner {
  padding: 12px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.5;
  background: rgba(245, 158, 11, 0.12);
  color: #b45309;
  border: 1px solid rgba(245, 158, 11, 0.3);
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
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
  margin-bottom: 6px;
}

.stat-label {
  font-size: 12px;
  color: var(--color-text-muted);
}

.stat-card--busy .stat-value {
  color: var(--status-running);
}

.stat-card--ok .stat-value {
  color: var(--status-done);
}

.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
}

.card-title {
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 10px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  list-style: none;
  margin: 0 0 4px;
  padding: 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.legend li {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

.dot--ring {
  background: transparent;
  border: 3px solid #f97316;
}

.runs {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.run {
  display: grid;
  grid-template-columns: auto minmax(120px, 1.2fr) auto minmax(160px, 2fr);
  align-items: center;
  gap: 10px;
  font-size: 13px;
}

.swatch {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  border: 4px solid;
}

.run__name {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.run__strategy,
.run__where {
  color: var(--color-text-muted);
  font-size: 12px;
}

.empty-state {
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
  margin: 0;
}

@media (max-width: 720px) {
  .run {
    grid-template-columns: auto 1fr;
  }

  .run__strategy,
  .run__where {
    grid-column: 2;
  }
}
</style>
