<template>
  <div class="infra">
    <div class="page-header">
      <div>
        <h1 class="page-title">Infrastructure</h1>
        <p class="subtitle">
          Hosts and Cooja containers available to run simulations.
          <span v-if="lastUpdated" class="when">Updated {{ lastUpdated }}.</span>
        </p>
      </div>
      <div class="header-actions">
        <label class="auto">
          <input v-model="autoRefresh" type="checkbox" />
          <span>Auto</span>
        </label>
        <button class="refresh-btn" :disabled="loading" @click="load()">
          {{ loading ? "Loading…" : "Refresh" }}
        </button>
        <button class="primary-btn" @click="openCreate">Register host</button>
      </div>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>
    <p v-if="notice" class="notice-banner">{{ notice }}</p>

    <!-- §7.1 Overview -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">{{ totals.hosts }}</div>
        <div class="stat-label">
          Hosts
          <span v-if="totals.hosts_operational !== totals.hosts" class="of">
            · {{ totals.hosts_operational }} operational
          </span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ totals.resources }}</div>
        <div class="stat-label">Containers</div>
      </div>
      <div class="stat-card stat-card--free">
        <div class="stat-value">{{ totals.free }}</div>
        <div class="stat-label">Free</div>
      </div>
      <!--
        "In use" rather than "Running": `active` includes reservations that
        have not started a simulation yet, and the two are worth telling
        apart when the queue is moving.
      -->
      <div class="stat-card stat-card--active">
        <div class="stat-value">{{ totals.active }}</div>
        <div class="stat-label">
          In use
          <span v-if="activeDetail" class="of">· {{ activeDetail }}</span>
        </div>
      </div>
      <div class="stat-card stat-card--unavailable">
        <div class="stat-value">{{ totals.unavailable }}</div>
        <div class="stat-label">Unavailable</div>
      </div>
      <div class="stat-card stat-card--disabled">
        <div class="stat-value">{{ totals.disabled }}</div>
        <div class="stat-label">Disabled</div>
      </div>
    </div>

    <!-- A mixed pool can introduce a host effect in the results, so it is
         surfaced rather than left for someone to notice in the data. -->
    <p v-if="overview?.heterogeneous" class="warn-banner">
      This pool runs {{ overview.cooja_builds.length }} different Cooja builds. A generation
      distributed across them can pick up a host effect; the build of each container is in its row
      below.
    </p>

    <!-- Cloned VM images keep the source's host keys. The pin is what tells
         one host from another, so with a shared key a port forward pointing
         at the wrong VM would be accepted without a word. -->
    <p
      v-for="group in overview?.shared_host_keys ?? []"
      :key="group.fingerprint"
      class="warn-banner"
    >
      {{ group.hosts.join(", ") }} answer with the same SSH host key (<code>{{
        group.fingerprint
      }}</code
      >) — cloned images that kept their keys. The pinned key cannot tell them apart. On each
      machine run
      <code
        >sudo rm /etc/ssh/ssh_host_* &amp;&amp; sudo ssh-keygen -A &amp;&amp; sudo systemctl restart
        ssh</code
      >, then clear each host's pin so the new key is learned on the next check:
      <code>PUT /api/v1/infra/hosts/{id}</code> with
      <code>{"access": {"host_key_fingerprint": null}}</code> (master-node README, "Several hosts
      share one host key").
    </p>

    <MachinesCard v-if="machines.length" :machines="machines" @saved="load(false)" />

    <p v-if="!loading && hosts.length === 0" class="empty-state">
      No hosts registered yet. The local host is created by the master-node on startup — if it is
      missing, check that the master-node is running.
    </p>

    <!-- §7.3 Per-host occupancy -->
    <section v-for="host in hosts" :key="host.id" class="host-section">
      <HostCard
        :host="host"
        :occupancy="occupancyFor(host.id)"
        :pending="pendingHosts.has(host.id)"
        @check="onCheck"
        @discover="onDiscover"
        @toggle="onToggleHost"
        @edit="openEdit"
        @add-worker="openWorkerForm"
        @remove="onRemove"
      />
      <details
        class="containers"
        :open="expanded.has(host.id)"
        @toggle="onToggleDetails(host.id, $event)"
      >
        <summary>{{ resourcesFor(host.id).length }} container(s)</summary>
        <ResourceTable
          :resources="resourcesFor(host.id)"
          :busy-ids="pendingResources"
          :can-discover="host.discovery?.mode !== 'manual'"
          @toggle="onToggleResource"
          @edit="(worker) => openWorkerForm(host, worker)"
          @remove="onRemoveWorker"
        />
      </details>
    </section>

    <HostFormModal v-if="formOpen" :host="editing" @close="closeForm" @saved="onSaved" />

    <WorkerFormModal
      v-if="workerHost"
      :host="workerHost"
      :worker="editingWorker"
      @close="closeWorkerForm"
      @saved="onWorkerSaved"
    />
  </div>
</template>

<script setup lang="ts">
import "../components/infra/infra.css";
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import HostCard from "../components/infra/HostCard.vue";
import HostFormModal from "../components/infra/HostFormModal.vue";
import MachinesCard from "../components/infra/MachinesCard.vue";
import ResourceTable from "../components/infra/ResourceTable.vue";
import WorkerFormModal from "../components/infra/WorkerFormModal.vue";
import { confirmDialog } from "../composables/useConfirm";
import {
  deleteHost,
  deleteResource,
  getOverview,
  listHosts,
  listMachines,
  listResources,
  requestCheck,
  requestDiscovery,
  setHostEnabled,
  setResourceEnabled,
} from "../api/infra";
import type {
  HostDto,
  MachineDto,
  OverviewDto,
  OverviewTotalsDto,
  ResourceDto,
} from "../types/infra";

/**
 * The Infrastructure page.
 *
 * Two behaviours are worth knowing when reading this:
 *
 *  - **Check and rediscover are requests, not actions.** The API records them
 *    and answers 202; the master-node carries them out, because it is the only
 *    process with a route to the containers and with the key to open a stored
 *    credential. So those buttons show a pending state and the page polls for
 *    the result rather than reading it from the response.
 *  - **Polling pauses while a form is open.** A refresh landing mid-edit
 *    would replace the host under the operator's cursor.
 */
const EMPTY_TOTALS: OverviewTotalsDto = {
  hosts: 0,
  hosts_operational: 0,
  resources: 0,
  free: 0,
  active: 0,
  unavailable: 0,
  disabled: 0,
};

const POLL_MS = 5000;

const hosts = ref<HostDto[]>([]);
const resources = ref<ResourceDto[]>([]);
const machines = ref<MachineDto[]>([]);
const overview = ref<OverviewDto | null>(null);
const loading = ref(false);
const error = ref("");
const notice = ref("");
const lastUpdated = ref("");
const autoRefresh = ref(true);

const formOpen = ref(false);
const editing = ref<HostDto | null>(null);
const workerHost = ref<HostDto | null>(null);
const editingWorker = ref<ResourceDto | null>(null);
const pendingHosts = reactive(new Set<string>());
const pendingResources = reactive(new Set<string>());
const expanded = reactive(new Set<string>());

const totals = computed(() => overview.value?.totals ?? EMPTY_TOTALS);

/**
 * What the containers in use are actually doing.
 *
 * `active` is reservations plus running simulations. While the queue moves
 * the two differ, and the card used to call all of it "Running".
 */
const activeDetail = computed(() => {
  const running = totals.value.running;
  const reserved = totals.value.reserved;
  if (running === undefined || reserved === undefined) return "";
  if (!reserved) return "";
  return `${running} running, ${reserved} starting`;
});

function occupancyFor(hostId: string) {
  return overview.value?.hosts.find((h) => h.id === hostId) ?? null;
}

function resourcesFor(hostId: string): ResourceDto[] {
  return resources.value.filter((r) => r.host_id === hostId);
}

let poll: ReturnType<typeof setInterval> | null = null;
let inFlight: AbortController | null = null;

async function load(showSpinner = true) {
  if (showSpinner) loading.value = true;
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  try {
    const [nextOverview, nextHosts, nextResources, nextMachines] = await Promise.all([
      getOverview(controller.signal),
      listHosts(controller.signal),
      listResources(controller.signal),
      listMachines(controller.signal),
    ]);
    overview.value = nextOverview;
    hosts.value = nextHosts;
    resources.value = nextResources;
    machines.value = nextMachines;
    lastUpdated.value = new Date().toLocaleTimeString();
    error.value = "";
  } catch (e) {
    if (!controller.signal.aborted) error.value = describe(e);
  } finally {
    if (inFlight === controller) inFlight = null;
    loading.value = false;
  }
}

onMounted(async () => {
  await load();
  poll = setInterval(() => {
    // Not while a form is open: a refresh would replace the record being
    // edited under the operator's cursor.
    if (autoRefresh.value && !formOpen.value && !workerHost.value) load(false);
  }, POLL_MS);
});

onBeforeUnmount(() => {
  if (poll) clearInterval(poll);
  inFlight?.abort();
});

// ── Requests the master-node carries out ─────────────────────────────────────

async function onCheck(host: HostDto) {
  await withPending(pendingHosts, host.id, async () => {
    const accepted = await requestCheck(host.id);
    notice.value = `${host.name}: ${accepted.detail}`;
  });
}

async function onDiscover(host: HostDto) {
  await withPending(pendingHosts, host.id, async () => {
    const accepted = await requestDiscovery(host.id);
    notice.value = `${host.name}: ${accepted.detail}`;
  });
}

// ── Pool membership ──────────────────────────────────────────────────────────

async function onToggleHost(host: HostDto, enabled: boolean) {
  if (!enabled) {
    const occupancy = occupancyFor(host.id);
    const running = occupancy?.active ?? 0;
    const ok = await confirmDialog({
      title: `Take ${host.name} out of the pool?`,
      message: running
        ? `${running} simulation(s) are running on it. They will be left to ` +
          `finish — disabling only stops new ones being placed here.`
        : "No new simulations will be placed on this host.",
      confirmLabel: "Disable",
    });
    if (!ok) return;
  }
  await withPending(pendingHosts, host.id, async () => {
    await setHostEnabled(host.id, enabled);
  });
}

async function onToggleResource(resource: ResourceDto, enabled: boolean) {
  await withPending(pendingResources, resource.id, async () => {
    await setResourceEnabled(resource.id, enabled);
  });
}

async function onRemove(host: HostDto) {
  const ok = await confirmDialog({
    title: `Remove ${host.name}?`,
    message:
      "Its containers are removed from the inventory too. Simulations that " +
      "ran on them keep the host and container names they recorded, so the " +
      "trail is not lost.",
    confirmLabel: "Remove",
    danger: true,
  });
  if (!ok) return;

  await withPending(pendingHosts, host.id, async () => {
    try {
      await deleteHost(host.id);
      notice.value = `${host.name} removed.`;
    } catch (e) {
      // Preserve the server's distinction between active work, retained
      // results and a concurrent registration. Each needs a different action.
      const message = describe(e);
      error.value = message;
      throw new Error(message);
    }
  });
}

// ── Form ─────────────────────────────────────────────────────────────────────

function openCreate() {
  editing.value = null;
  formOpen.value = true;
}

function openEdit(host: HostDto) {
  editing.value = host;
  formOpen.value = true;
}

function closeForm() {
  formOpen.value = false;
  editing.value = null;
}

async function onSaved(id: string) {
  closeForm();
  notice.value =
    "Host saved. The master-node will probe it and look for Cooja " +
    "containers on its next cycle.";
  expanded.add(id);
  await load(false);
}

// ── Workers ──────────────────────────────────────────────────────────────────

function openWorkerForm(host: HostDto, worker?: ResourceDto) {
  workerHost.value = host;
  editingWorker.value = worker ?? null;
}

function closeWorkerForm() {
  workerHost.value = null;
  editingWorker.value = null;
}

async function onWorkerSaved(id: string) {
  const hostId = workerHost.value?.id;
  closeWorkerForm();
  notice.value =
    "Worker saved. The master-node probes it on the next cycle; it joins the " +
    "pool once its sshd answers and a Cooja build is found.";
  if (hostId) expanded.add(hostId);
  void id;
  await load(false);
}

async function onRemoveWorker(worker: ResourceDto) {
  const ok = await confirmDialog({
    title: `Remove ${worker.name}?`,
    message:
      "Simulations that ran on it keep the names they recorded, so the trail " + "is not lost.",
    confirmLabel: "Remove",
    danger: true,
  });
  if (!ok) return;

  await withPending(pendingResources, worker.id, async () => {
    try {
      await deleteResource(worker.id);
    } catch (e) {
      const message = describe(e);
      // A retained result also refuses removal; waiting for a run to finish
      // cannot resolve it. Keep the actionable diagnosis from the API.
      error.value = message;
      throw new Error(message);
    }
  });
}

// ── helpers ──────────────────────────────────────────────────────────────────

function onToggleDetails(hostId: string, event: Event) {
  const open = (event.target as HTMLDetailsElement).open;
  if (open) expanded.add(hostId);
  else expanded.delete(hostId);
}

async function withPending(set: Set<string>, id: string, action: () => Promise<void>) {
  set.add(id);
  notice.value = "";
  try {
    await action();
    await load(false);
  } catch (e) {
    if (!error.value) error.value = describe(e);
  } finally {
    set.delete(id);
  }
}

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
</script>

<style scoped>
.infra {
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

.refresh-btn,
.primary-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}

.refresh-btn {
  background: var(--color-primary-light);
  color: var(--color-link, var(--color-primary));
  border: 1px solid #bfdbfe;
}

.primary-btn {
  background: var(--color-primary);
  color: #fff;
  border: 1px solid var(--color-primary);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.error-banner,
.notice-banner,
.warn-banner {
  padding: 12px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.5;
}

.error-banner {
  background: #fee2e2;
  color: var(--status-error);
  border: 1px solid #fecaca;
}

.notice-banner {
  background: var(--color-primary-light);
  color: var(--color-link, var(--color-primary));
  border: 1px solid #bfdbfe;
}

.warn-banner {
  background: rgba(245, 158, 11, 0.12);
  color: #b45309;
  border: 1px solid rgba(245, 158, 11, 0.3);
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
}

.stat-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
}

.stat-value {
  font-size: 32px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 6px;
}

.stat-label {
  font-size: 12px;
  color: var(--color-text-muted);
  font-weight: 500;
}

.of {
  font-weight: 400;
}

.stat-card--free .stat-value {
  color: var(--status-done);
}
.stat-card--active .stat-value {
  color: var(--status-running);
}
.stat-card--unavailable .stat-value {
  color: var(--status-error);
}
.stat-card--disabled .stat-value {
  color: var(--status-cancelled);
}

.host-section {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.containers {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-top: none;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
  margin-top: -6px;
  padding: 0 4px 4px;
}

.containers summary {
  cursor: pointer;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.containers summary:hover {
  color: var(--color-text);
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: var(--color-text-muted);
  font-style: italic;
  font-size: 13px;
  line-height: 1.6;
}
</style>
