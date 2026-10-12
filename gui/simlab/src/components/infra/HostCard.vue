<template>
  <article class="host-card" :class="{ 'host-card--out': !inPool }">
    <header>
      <div class="identity">
        <h3>
          {{ host.name }}
          <span class="type">{{ host.type }}</span>
        </h3>
        <p class="where">
          <template v-if="host.access">
            <span class="mono">{{ host.access.username }}@{{ host.access.address }}:{{ host.access.port }}</span>
            <span class="sep">·</span>
            containers via {{ addressModeLabel }}
          </template>
          <template v-else>
            this machine's own Docker daemon
          </template>
        </p>
      </div>

      <div class="status-block">
        <span :class="['status', `status--${host.status}`]">{{ host.status }}</span>
        <span v-if="!host.enabled" class="status status--disabled">out of pool</span>
      </div>
    </header>

    <p v-if="host.last_check_message" class="message">
      {{ host.last_check_message }}
      <span v-if="host.last_check" class="when">· {{ relative(host.last_check) }}</span>
    </p>
    <p v-else class="message message--muted">Not checked yet.</p>

    <!-- A host that cannot be reached still reports the capacity it has, so
         the bar shows containers even while nothing can be placed on them. -->
    <OccupancyBar
      v-if="occupancy"
      :free="occupancy.free"
      :active="occupancy.active"
      :unavailable="occupancy.unavailable"
      :disabled="occupancy.disabled"
    />

    <p v-if="!schedulable && hasCapacity" class="warn">
      {{ occupancy?.free }} container(s) are free but nothing will be placed here:
      <template v-if="!host.enabled">the host is out of the pool.</template>
      <template v-else>its status is “{{ host.status }}”.</template>
    </p>

    <footer>
      <div class="facts">
        <span v-if="host.discovery?.discovered_count !== undefined">
          {{ host.discovery.discovered_count }} discovered
        </span>
        <span v-if="host.consecutive_failures > 0" class="failing">
          {{ host.consecutive_failures }} consecutive failure(s)
        </span>
        <span v-if="host.access?.host_key_fingerprint" :title="host.access.host_key_fingerprint">
          host key pinned
        </span>
      </div>

      <div class="actions">
        <button
          class="ghost"
          :disabled="pending"
          :title="checkTitle"
          @click="emit('check', host)"
        >
          {{ host.check_pending ? "Checking…" : "Test" }}
        </button>
        <button
          class="ghost"
          :disabled="pending"
          title="Ask the master-node to look for Cooja containers on this host"
          @click="emit('discover', host)"
        >
          {{ host.discovery_pending ? "Discovering…" : "Rediscover" }}
        </button>
        <button class="ghost" :disabled="pending" @click="emit('toggle', host, !host.enabled)">
          {{ host.enabled ? "Disable" : "Enable" }}
        </button>
        <button v-if="editable" class="ghost" :disabled="pending" @click="emit('edit', host)">
          Edit
        </button>
        <button
          v-if="editable"
          class="ghost ghost--danger"
          :disabled="pending"
          @click="emit('remove', host)"
        >
          Remove
        </button>
      </div>
    </footer>
  </article>
</template>

<script setup lang="ts">
import { computed } from "vue";
import OccupancyBar from "./OccupancyBar.vue";
import type { HostDto, HostOccupancyDto } from "../../types/infra";

/**
 * One host: how it is reached, what was measured, and what is on it.
 *
 * The local host is shown but not editable: it is created by the master-node
 * from its own environment, so there is nothing here for an operator to
 * configure, and removing it would only have it reappear on the next start.
 */
const props = defineProps<{
  host: HostDto;
  occupancy?: HostOccupancyDto | null;
  pending?: boolean;
}>();

const emit = defineEmits<{
  (event: "check", host: HostDto): void;
  (event: "discover", host: HostDto): void;
  (event: "toggle", host: HostDto, enabled: boolean): void;
  (event: "edit", host: HostDto): void;
  (event: "remove", host: HostDto): void;
}>();

const editable = computed(() => props.host.type !== "local");
const schedulable = computed(() => props.occupancy?.schedulable ?? false);
const hasCapacity = computed(() => (props.occupancy?.free ?? 0) > 0);
const inPool = computed(() => props.host.enabled && schedulable.value);

const addressModeLabel = computed(() => {
  switch (props.host.container_access?.address_mode) {
    case "container_ip":
      return "bridge IP";
    case "published_port":
      return "published port";
    case "container_name":
      return "Docker DNS";
    default:
      return "—";
  }
});

const checkTitle = computed(() =>
  "The master-node performs the check: it holds the key and the route to the" +
  " containers, which the API does not",
);

function relative(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(iso).toLocaleString();
}
</script>

<style scoped>
.host-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.host-card--out { opacity: 0.75; }

header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

h3 {
  font-size: 15px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}

.type {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
  background: var(--color-surface-hover);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.where {
  font-size: 12px;
  color: var(--color-text-muted);
  margin-top: 2px;
}

.sep { margin: 0 4px; }

.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

.status-block {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}

.status {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--color-surface-hover);
  color: var(--color-text-muted);
  white-space: nowrap;
}

.status--online { background: rgba(16, 185, 129, 0.14); color: var(--status-done); }
.status--degraded { background: rgba(245, 158, 11, 0.16); color: var(--status-waiting); }
.status--offline,
.status--misconfigured { background: rgba(239, 68, 68, 0.12); color: var(--status-error); }

.message {
  font-size: 12px;
  color: var(--color-text);
}

.message--muted { color: var(--color-text-muted); font-style: italic; }

.when { color: var(--color-text-muted); }

.warn {
  font-size: 12px;
  padding: 8px 10px;
  border-radius: var(--radius-md);
  background: rgba(245, 158, 11, 0.12);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: #b45309;
}

footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  border-top: 1px solid var(--color-border);
  padding-top: 10px;
}

.facts {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 11px;
  color: var(--color-text-muted);
}

.failing { color: var(--status-error); font-weight: 600; }

.actions { display: flex; gap: 6px; flex-wrap: wrap; }

.ghost {
  padding: 5px 10px;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.12s;
}

.ghost:hover:not(:disabled) { background: var(--color-surface-hover); }
.ghost:disabled { opacity: 0.5; cursor: default; }

.ghost--danger { color: var(--status-error); border-color: rgba(239, 68, 68, 0.35); }
</style>
