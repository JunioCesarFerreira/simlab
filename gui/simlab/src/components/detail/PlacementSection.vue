<template>
  <div class="card pl-card">
    <div class="section-title pl-title">
      <span>
        Work distribution
        <span v-if="data && data.placed" class="pl-count">
          {{ data.placed }} simulation{{ data.placed === 1 ? "" : "s" }} across
          {{ containers }} container{{ containers === 1 ? "" : "s" }}
        </span>
      </span>
      <button v-if="!loading && !error" class="pl-btn" @click="load">Refresh</button>
    </div>

    <p v-if="loading" class="pl-note">Loading…</p>
    <p v-else-if="error" class="pl-note pl-note--error">{{ error }}</p>
    <p v-else-if="!data || !data.placed" class="pl-note">{{ emptyMessage(data) }}</p>

    <template v-else>
      <!-- Shares are of what was placed, so a run still waiting does not
           silently shrink the denominator. -->
      <p v-if="data.unplaced" class="pl-note pl-note--warn">
        {{ data.unplaced }} of {{ data.total }} simulation{{ data.total === 1 ? "" : "s" }} never
        reached a container, and {{ data.unplaced === 1 ? "is" : "are" }} not counted in the
        percentages below.
      </p>

      <div v-for="host in data.hosts" :key="host.host_id || host.host_name" class="pl-host">
        <div class="pl-host-head">
          <span class="pl-host-name">{{ host.host_name }}</span>
          <span class="pl-host-share">{{ formatShare(host.share) }}</span>
          <span class="pl-host-count">{{ host.total }}</span>
        </div>

        <div
          v-for="worker in host.workers"
          :key="worker.resource_id || worker.resource_name"
          class="pl-worker"
        >
          <span class="pl-worker-name" :title="worker.resource_name">
            {{ worker.resource_name }}
          </span>
          <div
            class="pl-bar"
            role="img"
            :aria-label="`${worker.resource_name}: ${describeOutcome(worker)}`"
          >
            <!-- One bar, split by outcome: a container that took work and
                 produced nothing is the case a single number cannot show. -->
            <div
              v-if="worker.done"
              class="pl-seg pl-seg--done"
              :style="{ width: `${widthOf(worker.done)}%` }"
            />
            <div
              v-if="worker.failed"
              class="pl-seg pl-seg--failed"
              :style="{ width: `${widthOf(worker.failed)}%` }"
            />
            <div
              v-if="worker.active"
              class="pl-seg pl-seg--active"
              :style="{ width: `${widthOf(worker.active)}%` }"
            />
          </div>
          <span class="pl-worker-share">{{ formatShare(worker.share) }}</span>
          <span class="pl-worker-count">{{ describeOutcome(worker) }}</span>
        </div>
      </div>

      <p class="pl-legend">
        <span class="pl-key pl-key--done" />done <span class="pl-key pl-key--failed" />failed
        <span class="pl-key pl-key--active" />running
        <span class="pl-legend-note">
          Counted per simulation, so a replica that was repeated counts on both containers that ran
          it.
        </span>
      </p>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { getExperimentPlacement } from "../../api/experiments";
import type { PlacementDistributionDto } from "../../types/simlab";
import {
  busiestWorker,
  describeOutcome,
  emptyMessage,
  formatShare,
  segmentWidth,
  workerCount,
} from "../../services/placement";

const props = defineProps<{ experimentId: string }>();

const data = ref<PlacementDistributionDto | null>(null);
const loading = ref(false);
const error = ref("");

const containers = computed(() => workerCount(data.value));
// The busiest container sets the bar scale, so the comparison is between
// containers rather than against a total nobody reaches.
const busiest = computed(() => busiestWorker(data.value));

function widthOf(count: number): number {
  return segmentWidth(count, busiest.value);
}

async function load(): Promise<void> {
  if (!props.experimentId) return;
  loading.value = true;
  error.value = "";
  try {
    data.value = await getExperimentPlacement(props.experimentId);
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

watch(() => props.experimentId, load, { immediate: true });
</script>

<style scoped>
.pl-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.pl-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.pl-count {
  margin-left: 8px;
  font-size: 12px;
  font-weight: 400;
  color: var(--color-text-muted);
}

.pl-btn {
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--color-border);
  background: var(--color-bg);
  color: var(--color-text);
  cursor: pointer;
}

.pl-btn:hover {
  background: var(--color-surface);
}

.pl-note {
  margin: 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.pl-note--warn {
  color: var(--status-warning, #b45309);
}

.pl-note--error {
  color: var(--status-error);
}

.pl-host + .pl-host {
  margin-top: 14px;
}

.pl-host-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--color-border);
}

.pl-host-name {
  font-weight: 600;
  font-size: 13px;
}

.pl-host-share {
  font-size: 12px;
  color: var(--color-text-muted);
}

.pl-host-count {
  margin-left: auto;
  font-size: 11px;
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
}

.pl-worker {
  display: grid;
  grid-template-columns: minmax(80px, 150px) 1fr 56px auto;
  align-items: center;
  gap: 10px;
  padding: 5px 0;
  font-size: 12px;
}

.pl-worker-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono, ui-monospace, monospace);
}

.pl-bar {
  display: flex;
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--color-bg);
}

.pl-seg--done {
  background: var(--status-done, #15803d);
}

.pl-seg--failed {
  background: var(--status-error, #dc2626);
}

.pl-seg--active {
  background: var(--status-running, #2563eb);
}

.pl-worker-share {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.pl-worker-count {
  color: var(--color-text-muted);
  font-variant-numeric: tabular-nums;
}

.pl-legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  font-size: 11px;
  color: var(--color-text-muted);
}

.pl-key {
  width: 10px;
  height: 10px;
  border-radius: 3px;
}

.pl-key + .pl-key {
  margin-left: 8px;
}

.pl-key--done {
  background: var(--status-done, #15803d);
}

.pl-key--failed {
  background: var(--status-error, #dc2626);
}

.pl-key--active {
  background: var(--status-running, #2563eb);
}

.pl-legend-note {
  margin-left: auto;
  text-align: right;
}

@media (max-width: 640px) {
  .pl-worker {
    grid-template-columns: minmax(70px, 1fr) 48px;
    grid-template-areas:
      "name share"
      "bar  bar"
      "count count";
  }

  .pl-worker-name {
    grid-area: name;
  }

  .pl-bar {
    grid-area: bar;
  }

  .pl-worker-share {
    grid-area: share;
  }

  .pl-worker-count {
    grid-area: count;
  }

  .pl-legend-note {
    display: none;
  }
}
</style>
