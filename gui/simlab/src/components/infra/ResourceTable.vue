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
            <RouterLink
              v-if="resource.current_simulation_id"
              :to="simulationLink(resource)"
              class="sim-link"
            >
              {{ resource.current_simulation_id.slice(-8) }}
            </RouterLink>
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
  </div>
</template>

<script setup lang="ts">
import { RouterLink } from "vue-router";
import {
  dockerLevel,
  endpointLabel,
  overrideSummary,
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

const busyIds = props.busyIds ?? new Set<string>();

function probeDetail(resource: ResourceDto): string {
  const message = resource.probe?.message ?? "";
  // "ready" is already the label; repeating it below adds nothing.
  return message && message !== "ready" ? message : "";
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

function simulationLink(resource: ResourceDto): string {
  // The simulation itself has no page; its experiment does, and that is
  // where an operator wants to land.
  return `/experiments?simulation=${resource.current_simulation_id ?? ""}`;
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

.right { text-align: right; }
.nowrap { white-space: nowrap; }

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

.mini:hover:not(:disabled) { background: var(--color-surface-hover); }
.mini:disabled { opacity: 0.5; cursor: default; }
.mini--danger { color: var(--status-error); border-color: rgba(239, 68, 68, 0.35); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

.name { font-weight: 600; }

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

.pill--available { background: rgba(16, 185, 129, 0.14); color: var(--status-done); }
.pill--busy,
.pill--reserved { background: rgba(59, 130, 246, 0.14); color: var(--status-running); }
.pill--offline,
.pill--error { background: rgba(239, 68, 68, 0.12); color: var(--status-error); }

.level {
  font-size: 12px;
  font-weight: 500;
}

.level--ok { color: var(--status-done); }
.level--bad { color: var(--status-error); }
.level--unknown { color: var(--color-text-muted); font-style: italic; }

.sim-link {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--color-primary);
}

.sim-link:hover { text-decoration: underline; }

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
  transition: transform 0.15s, background 0.15s;
}

.toggle input:checked + .toggle-track {
  background: var(--color-primary-light);
  border-color: var(--color-primary);
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
