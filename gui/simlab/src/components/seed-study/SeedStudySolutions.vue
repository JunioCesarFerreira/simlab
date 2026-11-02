<template>
  <div class="solutions">
    <p v-if="error" class="muted">{{ error }}</p>
    <p v-else-if="!data" class="muted">Loading…</p>
    <p v-else-if="!data.solutions.length" class="muted">
      The panel is chosen when the study starts. Its solutions appear here then.
    </p>
    <template v-else>
      <div class="solutions__bar">
        <label class="solutions__pick">
          Solution
          <select v-model="index">
            <option v-for="(s, i) in data.solutions" :key="s.genome_hash" :value="i">
              {{ chromosomeLabel(s.genome_hash, i) }}
            </option>
          </select>
        </label>
        <button
          type="button"
          class="step-btn"
          :disabled="index === 0"
          aria-label="Previous solution"
          @click="index--"
        >
          ‹
        </button>
        <button
          type="button"
          class="step-btn"
          :disabled="index === data.solutions.length - 1"
          aria-label="Next solution"
          @click="index++"
        >
          ›
        </button>
        <ul class="facts">
          <li
            v-for="fact in scene.facts"
            :key="fact.label"
            :class="['fact', fact.tone && `fact--${fact.tone}`]"
            :title="fact.hint"
          >
            {{ fact.label }}
          </li>
        </ul>
        <ChartExportButton v-if="scene.kind !== 'none'" @click="exportPng" />
      </div>

      <p v-if="scene.kind === 'none'" class="muted">{{ scene.message }}</p>
      <template v-else>
        <div
          ref="el"
          class="solutions__canvas"
          :style="{ height: `${height}px` }"
          role="img"
          :aria-label="`Solution ${index + 1} of ${data.solutions.length}`"
        />
        <ResizeHandle v-model="height" label="solution chart" :min="260" :max="900" />
        <p class="note">{{ note }}</p>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { getStudySolutions } from "../../api/seedStudy";
import { useEChart } from "../../composables/useEChart";
import { usePersistedHeight } from "../../composables/usePersistedHeight";
import { useTheme } from "../../composables/useTheme";
import { chartExportBackground, chartPalette } from "../../services/chartTheme";
import { chromosomeLabel } from "../../services/seedStudy";
import { solutionScene, topologyOption, type SolutionScene } from "../../services/solutionTopology";
import type { StudyDetail, StudySolutions } from "../../types/seedStudy";
import { chartExportFilename } from "../../utils/chartExport";
import ChartExportButton from "../charts/ChartExportButton.vue";
import ResizeHandle from "../common/ResizeHandle.vue";

/**
 * The solutions of a study's panel, one at a time, drawn the way their problem
 * reads: relays, a selection of candidates, a route, or a decision vector.
 */
const props = defineProps<{ detail: StudyDetail }>();

const data = ref<StudySolutions | null>(null);
const error = ref<string | null>(null);
const index = ref(0);
const el = ref<HTMLElement | null>(null);
const chart = useEChart(el);
const { isDark } = useTheme();
const height = usePersistedHeight("seed-study-solution", 480);

let controller: AbortController | null = null;

async function load() {
  controller?.abort();
  controller = new AbortController();
  error.value = null;
  try {
    data.value = await getStudySolutions(props.detail.id, controller.signal);
    if (index.value >= data.value.solutions.length) index.value = 0;
  } catch (e) {
    if ((e as { name?: string }).name === "CanceledError") return;
    error.value = "Could not load the solutions of this study.";
  }
}

watch(
  () => props.detail.id,
  () => {
    data.value = null;
    index.value = 0;
    load();
  },
  { immediate: true },
);
// The panel is recorded once the engine starts the study; read it again then.
watch(
  () => props.detail.panel.length,
  (size) => {
    if (data.value && data.value.solutions.length < size) load();
  },
);
onBeforeUnmount(() => controller?.abort());

const scene = computed<SolutionScene>(() => {
  const solution = data.value?.solutions[index.value];
  if (!data.value || !solution) return { kind: "none", message: "", facts: [] };
  return solutionScene(data.value.problem, solution.chromosome);
});

// The axes keep one scale for x and y, which depends on the size of the canvas.
const size = ref({ width: 0, height: 0 });
let observer: ResizeObserver | null = null;
watch(el, (node) => {
  observer?.disconnect();
  observer = null;
  if (!node) return;
  observer = new ResizeObserver(() => {
    if (node.clientWidth > 0 && node.clientHeight > 0) {
      size.value = { width: node.clientWidth, height: node.clientHeight };
    }
  });
  observer.observe(node);
});
onBeforeUnmount(() => observer?.disconnect());

watch(
  [scene, size, isDark, chart.ready],
  ([current, { width, height: h }, dark, ready]) => {
    if (!ready || current.kind === "none" || width === 0) return;
    chart.setOption(topologyOption(current, chartPalette(dark), width, h), true);
  },
  { immediate: true },
);

const note = computed(() => {
  if (scene.value.kind === "vector") {
    return "An analytical benchmark: the solution is its decision vector, and there is no network.";
  }
  const parts = ["Drawn from the solution as recorded for the study, after its firmware was resolved."];
  if (scene.value.kind === "network") parts.push(scene.value.linksNote);
  if (props.detail.synthetic) parts.push("This study ran synthetically: its metrics did not come from this network.");
  return parts.join(" ");
});

function exportPng() {
  chart.exportImage(
    chartExportFilename(`seed-study-solution-${index.value + 1}`),
    { backgroundColor: chartExportBackground(isDark.value) },
  );
}
</script>

<style scoped>
.solutions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.solutions__bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.solutions__pick {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.solutions__pick select {
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 12px;
}

.step-btn {
  width: 26px;
  height: 26px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 14px;
  cursor: pointer;
}

.step-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.facts {
  list-style: none;
  margin: 0 auto 0 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.fact {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}

.fact--ok {
  color: var(--status-done);
}

.fact--warn {
  color: #b45309;
}

.solutions__canvas {
  width: 100%;
}

.note {
  margin: 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.muted {
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
}
</style>
