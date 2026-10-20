<template>
  <div class="resource-table">
    <table v-if="resources.length > 0">
      <thead>
        <tr>
          <th>Container</th>
          <th>Pool</th>
          <th>Docker</th>
          <th>SSH probe</th>
          <th>Reached at</th>
          <th>Running</th>
          <th class="right">In pool</th>
          <th class="right"></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="resource in resources" :key="resource.id">
          <td>
            <div class="name">{{ resource.name }}</div>
            <div v-if="resource.probe.java_version" class="sub" :title="buildTitle(resource)">
              {{ shortBuild(resource) }}
            </div>
          </td>

          <td>
            <span :class="['pill', `pill--${resource.status}`]">
              {{ resource.status }}
            </span>
          </td>

          <!-- Level 1 and level 2 are shown side by side on purpose: "the
               container is stopped" and "it is up but has no Cooja" need
               different fixes, and one combined badge would hide which. -->
          <td>
            <span
              :class="['level', `level--${dockerLevel(resource).tone}`]"
              :title="dockerLevel(resource).title"
            >
              {{ dockerLevel(resource).label }}
            </span>
          </td>
          <td>
            <span
              :class="['level', `level--${probeLevel(resource).tone}`]"
              :title="probeLevel(resource).title"
            >
              {{ probeLevel(resource).label }}
            </span>
            <div v-if="probeDetail(resource)" class="sub">
              {{ probeDetail(resource) }}
            </div>
          </td>

          <td class="sub">
            <span class="mono">{{ endpointLabel(resource) }}</span>
            <div v-if="overrideSummary(resource)" class="sub">
              {{ overrideSummary(resource) }}
            </div>
          </td>

          <td>
            <button
              v-if="resource.current_simulation_id"
              class="sim-link"
              :disabled="resolving === resource.id"
              :title="`Open the experiment this simulation belongs to`"
              @click="openSimulation(resource)"
            >
              {{ resource.current_simulation_id.slice(-8) }}
            </button>
            <span v-else class="sub">—</span>
          </td>

          <td class="right">
            <label class="toggle" :title="toggleTitle(resource)">
              <input
                type="checkbox"
                :checked="resource.enabled"
                :disabled="busyIds.has(resource.id)"
                @change="emit('toggle', resource, !resource.enabled)"
              />
              <span class="toggle-track"><span class="toggle-knob" /></span>
            </label>
          </td>

          <td class="right nowrap">
            <button
              class="mini"
              :disabled="busyIds.has(resource.id)"
              title="Edit this worker's address and SSH parameters"
              @click="emit('edit', resource)"
            >
              Edit
            </button>
            <button
              v-if="resource.origin === 'manual'"
              class="mini mini--danger"
              :disabled="busyIds.has(resource.id)"
              @click="emit('remove', resource)"
            >
              Remove
            </button>
          </td>
        </tr>
      </tbody>
    </table>

    <p v-else class="empty">
      No containers discovered on this host yet.
      <template v-if="canDiscover">Run a discovery to look for them.</template>
    </p>
    <p v-if="linkError" class="link-error">{{ linkError }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { getResourceSimulation } from "../../api/infra";
import {
  dockerLevel,
  endpointLabel,
  overrideSummary,
  evidenceAge,
  probeLevel,
} from "../../services/resourceLevels";
import type { ResourceDto } from "../../types/infra";

/**
 * One host's workers, with both availability levels side by side.
 *
 * How each level is read lives in `services/resourceLevels`, not here: it is
 * domain logic mirroring a decision the master-node makes, and keeping it in
 * a pure module is what lets it be tested against that decision.
 */
const props = defineProps<{
  resources: ResourceDto[];
  /** Ids with a request in flight, so their controls are held. */
  busyIds?: Set<string>;
  canDiscover?: boolean;
}>();

const emit = defineEmits<{
  (event: "toggle", resource: ResourceDto, enabled: boolean): void;
  (event: "edit", resource: ResourceDto): void;
  (event: "remove", resource: ResourceDto): void;
}>();

const router = useRouter();
/** The worker whose simulation is being resolved, so its control is held. */
const resolving = ref<string | null>(null);
const linkError = ref("");

const busyIds = props.busyIds ?? new Set<string>();

function probeDetail(resource: ResourceDto): string {
  const message = resource.probe?.message ?? "";
  // "ready" is already the label. What is worth saying instead is *when* it
  // was measured: a healthy worker is only re-probed every few minutes, so a
  // worker that has stopped answering reads `ready` until the next one is due.
  if (message && message !== "ready") return message;
  return evidenceAge(resource.probe?.checked_at);
}

function shortBuild(resource: ResourceDto): string {
  const sha = resource.probe?.cooja_jar_sha256;
  return sha ? `cooja ${sha.slice(0, 8)}` : "";
}

function buildTitle(resource: ResourceDto): string {
  const parts = [resource.probe?.java_version, resource.container?.image].filter(Boolean);
  if (resource.probe?.cooja_jar_sha256) {
    parts.push(`cooja.jar ${resource.probe.cooja_jar_sha256}`);
  }
  return parts.join("\n");
}

/**
 * Open the experiment a running simulation belongs to.
 *
 * The link used to be `/experiments?simulation=<id>`, which no page reads:
 * the click landed on the list of experiments with nothing resolved, and the
 * operator had to find the right one by hand. The simulation itself has no
 * page, but its experiment does — and the API already answers which one, so
 * the resolution happens here rather than being encoded in a URL nobody
 * interprets.
 */
async function openSimulation(resource: ResourceDto): Promise<void> {
  resolving.value = resource.id;
  try {
    const info = await getResourceSimulation(resource.id);
    if (info.experiment_id) {
      await router.push(`/experiments/${info.experiment_id}`);
      return;
    }
    // It finished between the page loading and the click. Saying so beats
    // navigating somewhere arbitrary.
    linkError.value = "That simulation is no longer running, so there is nothing to open.";
  } catch {
    linkError.value = "Could not find out which experiment that simulation is in.";
  } finally {
    resolving.value = null;
  }
}

function toggleTitle(resource: ResourceDto): string {
  if (!resource.enabled) return "Put this worker back in the pool";
  if (resource.status === "busy" || resource.status === "reserved") {
    return "Take it out of the pool; the simulation on it finishes first";
  }
  return "Take this worker out of the pool";
}
</script>

<style scoped>
.link-error {
  margin: 0.4rem 0 0;
  font-size: 0.8rem;
  color: var(--danger, #b91c1c);
}

button.sim-link {
  background: none;
  border: none;
  padding: 0;
  font: inherit;
  cursor: pointer;
}

button.sim-link:disabled {
  cursor: progress;
  opacity: 0.6;
}

.resource-table {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

th {
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
  padding: 6px 10px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--color-border);
  vertical-align: top;
}

tbody tr:last-child td {
  border-bottom: none;
}

.right {
  text-align: right;
}
.nowrap {
  white-space: nowrap;
}

.mini {
  padding: 3px 8px;
  margin-left: 4px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}

.mini:hover:not(:disabled) {
  background: var(--color-surface-hover);
}
.mini:disabled {
  opacity: 0.5;
  cursor: default;
}
.mini--danger {
  color: var(--status-error);
  border-color: rgba(239, 68, 68, 0.35);
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

.name {
  font-weight: 600;
}

.sub {
  font-size: 11px;
  color: var(--color-text-muted);
}

.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: capitalize;
  background: var(--color-surface-hover);
  color: var(--color-text-muted);
}

.pill--available {
  background: rgba(16, 185, 129, 0.14);
  color: var(--status-done);
}
.pill--busy,
.pill--reserved {
  background: rgba(59, 130, 246, 0.14);
  color: var(--status-running);
}
.pill--offline,
.pill--error {
  background: rgba(239, 68, 68, 0.12);
  color: var(--status-error);
}

.level {
  font-size: 12px;
  font-weight: 500;
}

.level--ok {
  color: var(--status-done);
}
.level--bad {
  color: var(--status-error);
}
.level--unknown {
  color: var(--color-text-muted);
  font-style: italic;
}

.sim-link {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--color-link, var(--color-primary));
}

.sim-link:hover {
  text-decoration: underline;
}

.toggle input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle {
  display: inline-flex;
  cursor: pointer;
}

.toggle-track {
  width: 34px;
  height: 18px;
  border-radius: 999px;
  background: var(--color-surface-hover);
  border: 1px solid var(--color-border);
  display: inline-flex;
  align-items: center;
  padding: 2px;
  transition: background 0.15s;
}

.toggle-knob {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: var(--color-text-muted);
  transition:
    transform 0.15s,
    background 0.15s;
}

.toggle input:checked + .toggle-track {
  background: var(--color-primary-light);
  border-color: var(--color-link, var(--color-primary));
}

.toggle input:checked + .toggle-track .toggle-knob {
  transform: translateX(16px);
  background: var(--color-primary);
}

.toggle input:disabled + .toggle-track {
  opacity: 0.5;
  cursor: default;
}

.empty {
  padding: 20px;
  text-align: center;
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
}
</style>
