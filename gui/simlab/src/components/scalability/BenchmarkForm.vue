<template>
  <form class="bench-form" @submit.prevent="submit">
    <div class="bench-form__grid">
      <label class="field">
        <span>Name</span>
        <input v-model.trim="name" type="text" required placeholder="e.g. pool after adding vm8" />
      </label>

      <label class="field">
        <span>Problem instance</span>
        <select v-model="problemId" required @change="onProblemChange">
          <option value="" disabled>Choose a saved problem…</option>
          <option v-for="p in problems" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
        <small v-if="!problems.length" class="hint">
          No saved problems — create one in <RouterLink to="/problems">Problems</RouterLink>.
        </small>
        <small v-else-if="workloadNote" class="hint">{{ workloadNote }}</small>
      </label>

      <fieldset class="field field--wide">
        <legend>Firmware per MAC protocol</legend>
        <div v-for="protocol in PROTOCOLS" :key="protocol" class="firmware-row">
          <span class="mono">{{ protocol }}</span>
          <select v-model="firmware[protocol]" :aria-label="`Source repository for ${protocol}`">
            <option value="">— not used —</option>
            <option v-for="r in repositories" :key="r.id" :value="r.id">{{ r.name }}</option>
          </select>
        </div>
        <small class="hint">
          With only one protocol mapped, the workload's MAC gene is set to it.
        </small>
      </fieldset>

      <label class="field">
        <span>Concurrency levels (N)</span>
        <div class="inline">
          <input
            v-model="levelsText"
            type="text"
            inputmode="numeric"
            :aria-invalid="!parsed.ok"
            aria-describedby="levels-help"
          />
          <button type="button" class="ghost-btn" :disabled="pool < 1" @click="resetLevels">
            Fit to pool
          </button>
        </div>
        <small id="levels-help" :class="['hint', { 'hint--error': !parsed.ok }]">
          {{
            parsed.ok
              ? `${pool} worker(s) can take work now. Each level runs as one wave of N identical simulations.`
              : parsed.error
          }}
        </small>
      </label>

      <label class="field">
        <span>Repetitions per level</span>
        <input v-model.number="repetitions" type="number" :min="1" :max="MAX_REPETITIONS" />
        <small class="hint">Order alternates up and down, so drift lands on every level.</small>
      </label>

      <label class="field">
        <span>Simulated duration (s)</span>
        <input
          v-model.number="duration"
          type="number"
          :min="DURATION_RANGE[0]"
          :max="DURATION_RANGE[1]"
        />
      </label>

      <label class="field">
        <span>Simulation seed</span>
        <input v-model.number="seed" type="number" min="0" />
        <small class="hint">One seed for every run: the work must be identical.</small>
      </label>
    </div>

    <p v-if="parsed.ok" class="cost">
      <strong>{{ cost.waves }}</strong> wave(s) ·
      <strong>{{ cost.simulations }}</strong> simulation(s)
      <span v-if="cost.beyondPool.length" class="cost__warn">
        · levels {{ cost.beyondPool.join(", ") }} exceed the {{ pool }} available workers: their
        waves will queue, and measure the pool at full occupancy rather than at that N.
      </span>
    </p>

    <p v-if="error" class="form-error" role="alert">{{ error }}</p>

    <div class="actions">
      <button type="submit" class="primary-btn" :disabled="!canSubmit">
        {{ submitting ? "Starting…" : "Start benchmark" }}
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { RouterLink } from "vue-router";
import { getProblem, listProblems, type ProblemInfoDto } from "../../api/problems";
import { getAllRepositories } from "../../api/repositories";
import { createBenchmark } from "../../api/scalability";
import { exportProblem } from "../../services/exportProblemJson";
import { exportChromosome } from "../../services/exportChromosomeJson";
import { defaultDataConversion } from "../../services/defaultDataConversion";
import {
  DURATION_RANGE,
  MAX_REPETITIONS,
  benchmarkCost,
  parseLevels,
  proposeLevels,
} from "../../services/usl";
import type { SourceRepositoryDto } from "../../types/simlab";
import type { BenchmarkCreated } from "../../types/scalability";

const props = defineProps<{
  /** Workers that can take work now — the pool the levels are proposed for. */
  pool: number;
}>();

const emit = defineEmits<{ created: [BenchmarkCreated] }>();

const PROTOCOLS = ["csma", "tsch"] as const;

const problems = ref<ProblemInfoDto[]>([]);
const repositories = ref<SourceRepositoryDto[]>([]);
const name = ref("");
const problemId = ref("");
const firmware = reactive<Record<string, string>>({ csma: "", tsch: "" });
const levelsText = ref(proposeLevels(props.pool).join(", "));
const repetitions = ref(3);
const duration = ref(60);
const seed = ref(123456);
const workloadNote = ref("");
const providedChromosome = ref<Record<string, unknown> | null>(null);
const submitting = ref(false);
const error = ref("");

const parsed = computed(() => parseLevels(levelsText.value));
const cost = computed(() =>
  parsed.value.ok
    ? benchmarkCost(parsed.value.levels, repetitions.value, props.pool)
    : { waves: 0, simulations: 0, beyondPool: [] },
);
const firmwareOptions = computed(() =>
  Object.fromEntries(Object.entries(firmware).filter(([, id]) => id)),
);
const canSubmit = computed(
  () =>
    !submitting.value &&
    name.value.length > 0 &&
    problemId.value.length > 0 &&
    parsed.value.ok &&
    Object.keys(firmwareOptions.value).length > 0 &&
    repetitions.value >= 1 &&
    repetitions.value <= MAX_REPETITIONS &&
    duration.value >= DURATION_RANGE[0] &&
    duration.value <= DURATION_RANGE[1],
);

function resetLevels() {
  levelsText.value = proposeLevels(props.pool).join(", ");
}

async function onProblemChange() {
  workloadNote.value = "";
  providedChromosome.value = null;
  try {
    const problem = await getProblem(problemId.value);
    if (!name.value) name.value = `scalability · ${problem.name}`;
    if (problem.draft.chromosome) {
      providedChromosome.value = exportChromosome(problem.draft.chromosome).chromosome as Record<
        string,
        unknown
      >;
      workloadNote.value = "Uses the chromosome defined in this problem.";
    } else {
      workloadNote.value =
        "No chromosome in this problem: the engine draws one feasible network and runs it in every wave.";
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function submit() {
  if (!canSubmit.value || !parsed.value.ok) return;
  submitting.value = true;
  error.value = "";
  try {
    // Exported the same way the experiment wizard does, so the benchmark runs
    // exactly the configuration an optimisation of this problem would.
    const problem = await getProblem(problemId.value);
    const exported = exportProblem(problem.draft);
    const created = await createBenchmark({
      name: name.value,
      problem: exported.problem as unknown as Record<string, unknown>,
      source_repository_options: firmwareOptions.value,
      levels: parsed.value.levels,
      repetitions: repetitions.value,
      duration: duration.value,
      seed: seed.value,
      chromosome: providedChromosome.value,
      data_conversion_config: defaultDataConversion() as unknown as Record<string, unknown>,
    });
    emit("created", created);
  } catch (e) {
    error.value = describe(e);
  } finally {
    submitting.value = false;
  }
}

function describe(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  return e instanceof Error ? e.message : String(e);
}

onMounted(async () => {
  try {
    const [p, r] = await Promise.all([listProblems(), getAllRepositories()]);
    problems.value = p;
    repositories.value = r;
    // One repository is the common case; preselect it rather than make
    // the form unusable until someone notices the empty select.
    const only = r.length === 1 ? r[0] : undefined;
    if (only) firmware.csma = only.id;
  } catch (e) {
    error.value = describe(e);
  }
});
</script>

<style scoped>
.bench-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.bench-form__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px 18px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 13px;
  border: none;
  padding: 0;
  margin: 0;
  min-width: 0;
}

.field > span,
.field legend {
  font-weight: 600;
  font-size: 12px;
  color: var(--color-text-muted);
}

.field--wide {
  grid-column: 1 / -1;
}

.field input,
.field select {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 13px;
  min-width: 0;
}

.field input[aria-invalid="true"] {
  border-color: var(--status-error);
}

.inline {
  display: flex;
  gap: 8px;
}

.inline input {
  flex: 1;
}

.firmware-row {
  display: grid;
  grid-template-columns: 56px 1fr;
  align-items: center;
  gap: 10px;
}

.mono {
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 12px;
}

.hint {
  font-size: 11px;
  color: var(--color-text-muted);
  line-height: 1.45;
}

.hint--error {
  color: var(--status-error);
}

.cost {
  margin: 0;
  font-size: 13px;
}

.cost__warn {
  color: #b45309;
}

.form-error {
  margin: 0;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  background: #fee2e2;
  color: var(--status-error);
  font-size: 13px;
}

.actions {
  display: flex;
  justify-content: flex-end;
}

.primary-btn,
.ghost-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.primary-btn {
  background: var(--color-primary);
  color: #fff;
  border: 1px solid var(--color-primary);
}

.ghost-btn {
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  white-space: nowrap;
}

.primary-btn:disabled,
.ghost-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
