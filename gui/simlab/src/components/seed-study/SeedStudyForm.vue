<template>
  <form class="study-form" @submit.prevent="submit">
    <div class="study-form__grid">
      <label class="field">
        <span>Name</span>
        <input v-model.trim="name" type="text" required placeholder="e.g. p1 · 30 seeds" />
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
      </label>

      <label class="field">
        <span>MAC protocol</span>
        <select v-model="protocol">
          <option v-for="p in PROTOCOLS" :key="p" :value="p">{{ p }}</option>
        </select>
        <small class="hint">
          One per study: mixing protocols would count the protocol as a difference between
          solutions. To optimise the MAC gene, run one study per protocol.
        </small>
      </label>

      <label class="field">
        <span>Firmware</span>
        <select v-model="firmware" required :aria-label="`Source repository for ${protocol}`">
          <option value="" disabled>Choose a repository…</option>
          <option v-for="r in repositories" :key="r.id" :value="r.id">{{ r.name }}</option>
        </select>
      </label>

      <label class="field">
        <span>Simulated duration (min)</span>
        <input
          v-model.number="duration"
          type="number"
          :min="DURATION_RANGE[0]"
          :max="DURATION_RANGE[1]"
        />
        <small class="hint">Every conclusion holds for this duration only.</small>
      </label>

      <fieldset class="field">
        <legend>Solutions</legend>
        <label class="radio">
          <input v-model="panelMode" type="radio" value="panel" />
          <span>
            Panel of
            <input
              v-model.number="panelSize"
              class="inline-number"
              type="number"
              min="1"
              :max="MAX_CHROMOSOMES"
              aria-label="Panel size"
              :disabled="panelMode !== 'panel'"
            />
            random solutions
          </span>
        </label>
        <label class="radio">
          <input v-model="panelMode" type="radio" value="chromosome" :disabled="!chromosome" />
          <span>This solution (the chromosome saved in the problem)</span>
        </label>
        <small
          v-if="panelMode === 'panel' && panelSize < MIN_PANEL"
          :class="['hint', 'hint--warn']"
        >
          Below {{ MIN_PANEL }} solutions, the number of seeds to rank them swings between panels.
          One solution answers how it varies, not how many seeds an optimisation needs.
        </small>
        <small v-else class="hint">
          How many seeds rank solutions depends on how much they differ, so that answer needs a
          panel. A random panel stands for the start of a search.
        </small>
      </fieldset>

      <fieldset class="field field--wide">
        <legend>Seeds</legend>
        <div class="seed-row">
          <label class="radio">
            <input v-model="seedMode" type="radio" value="generated" />
            <span>
              Generate
              <input
                v-model.number="seedCount"
                class="inline-number"
                type="number"
                :min="MIN_SEEDS"
                :max="MAX_SEEDS"
                aria-label="Number of seeds"
                :disabled="seedMode !== 'generated'"
              />
              seeds from master seed
              <input
                v-model.number="masterSeed"
                class="inline-number inline-number--wide"
                type="number"
                min="0"
                aria-label="Master seed"
                :disabled="seedMode !== 'generated'"
              />
            </span>
          </label>
          <label class="radio">
            <input v-model="seedMode" type="radio" value="imported" />
            <span>
              Import
              <input
                class="file-input"
                type="file"
                accept=".json,application/json"
                aria-label="Seeds file"
                :disabled="seedMode !== 'imported'"
                @change="onSeedFile"
              />
            </span>
          </label>
        </div>
        <small v-if="seedMode === 'generated'" class="hint">
          Distinct, apart in their low 16 bits, and stable by prefix: the first m of this list are
          what a smaller list would be.
        </small>
        <template v-else>
          <small v-if="fileError" class="hint hint--error">{{ fileError }}</small>
          <small v-for="e in seedCheck.errors" :key="e" class="hint hint--error">{{ e }}</small>
          <small v-if="importedSeeds.length && !seedCheck.errors.length" class="hint">
            ✓ {{ importedSeeds.length }} valid seeds
          </small>
          <small v-if="seedCheck.lowCollisions || seedCheck.closePairs" class="hint hint--warn">
            {{ seedCheck.lowCollisions }} pair(s) with the same low 16 bits and
            {{ seedCheck.closePairs }} close pair(s): the firmware only sees those bits, so these
            seeds are not fully independent.
          </small>
        </template>
      </fieldset>

      <fieldset class="field field--wide">
        <legend>Objectives</legend>
        <div v-for="(objective, i) in objectives" :key="i" class="objective-row">
          <select v-model="objective.metric_name" :aria-label="`Objective ${i + 1} metric`">
            <option v-for="m in metricNames" :key="m" :value="m">{{ m }}</option>
          </select>
          <select v-model="objective.goal" :aria-label="`Objective ${i + 1} goal`">
            <option value="min">min</option>
            <option value="max">max</option>
          </select>
          <select
            :value="aggregatorKind(objective.aggregator)"
            :aria-label="`Objective ${i + 1} aggregator`"
            @change="setAggregator(i, ($event.target as HTMLSelectElement).value)"
          >
            <option v-for="a in SEED_AGGREGATORS" :key="a.kind" :value="a.kind">
              {{ a.label }}
            </option>
          </select>
          <button
            type="button"
            class="ghost-btn"
            :disabled="objectives.length < 2"
            @click="objectives.splice(i, 1)"
          >
            Remove
          </button>
          <label v-if="aggregatorKind(objective.aggregator) === 'trimmed_mean'" class="trim">
            Trim per end
            <input
              :value="aggregatorTrim(objective.aggregator)"
              type="number"
              min="0"
              max="0.49"
              step="0.01"
              @input="setTrim(i, ($event.target as HTMLInputElement).value)"
            />
          </label>
        </div>
        <div>
          <button type="button" class="ghost-btn" @click="addObjective">Add objective</button>
        </div>
        <small class="hint">
          Use the aggregators the optimisation will use: the study evaluates that estimator.
        </small>
      </fieldset>

      <fieldset class="field field--wide">
        <legend>Targets</legend>
        <div class="target-row">
          <label>
            Confidence (%)
            <input v-model.number="confidencePct" type="number" min="80" max="99.9" step="any" />
          </label>
          <label>
            Relative error (%)
            <input v-model.number="errorPct" type="number" min="0.1" max="100" step="any" />
          </label>
          <label>
            Reliability
            <input v-model.number="reliability" type="number" min="0.5" max="0.99" step="any" />
          </label>
          <label>
            Control repeats
            <input
              v-model.number="controlReplicates"
              type="number"
              min="0"
              :max="MAX_CONTROL_REPLICATES"
            />
          </label>
        </div>
        <small class="hint">
          Targets can change later without simulating again. Control repeats run the first seed
          again on every solution, to verify the seed is the only source of variation.
        </small>
      </fieldset>
    </div>

    <p class="cost">
      <strong>{{ cost.total }}</strong> simulations ({{ cost.main }} + {{ cost.control }} control)
      <template v-if="hours !== null">
        · about <strong>{{ formatHours(hours) }}</strong> on {{ pool }} worker(s), from
        {{ estimate?.samples }} recent runs of this duration
      </template>
      <template v-else> · no recent run of this duration to estimate the time from</template>
      <span v-if="cost.overLimit" class="cost__warn">
        · over the limit of {{ MAX_SIMULATIONS }} simulations
      </span>
    </p>

    <p v-if="error" class="form-error" role="alert">{{ error }}</p>

    <div class="actions">
      <button type="submit" class="primary-btn" :disabled="!canSubmit">
        {{ submitting ? "Starting…" : "Start study" }}
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import { getProblem, listProblems, type ProblemInfoDto } from "../../api/problems";
import { getAllRepositories } from "../../api/repositories";
import { createStudy, getStudyEstimate } from "../../api/seedStudy";
import { parseSeedFile } from "../../lib/seedFile";
import {
  SEED_AGGREGATORS,
  aggregatorKind,
  aggregatorTrim,
  buildSeedAggregator,
  isValidSeedAggregator,
} from "../../lib/seedAggregator";
import { defaultDataConversion } from "../../services/defaultDataConversion";
import { exportChromosome } from "../../services/exportChromosomeJson";
import { exportProblem } from "../../services/exportProblemJson";
import {
  DEFAULT_CONTROL_REPLICATES,
  DEFAULT_DURATION,
  DEFAULT_PANEL,
  DEFAULT_SEEDS,
  DEFAULT_TARGETS,
  MAX_CHROMOSOMES,
  MAX_CONTROL_REPLICATES,
  MAX_SEEDS,
  MAX_SIMULATIONS,
  MIN_PANEL,
  MIN_SEEDS,
  checkSeeds,
  defaultObjectives,
  estimateHours,
  formatHours,
  studyCost,
} from "../../services/seedStudy";
import { DURATION_RANGE } from "../../services/usl";
import type { StudyCreated, StudyEstimate } from "../../types/seedStudy";
import type { ObjectiveItem, SeedAggregatorKind, SourceRepositoryDto } from "../../types/simlab";

const props = defineProps<{
  /** Workers that can take work now, for the time estimate. */
  pool: number;
}>();

const emit = defineEmits<{ created: [StudyCreated] }>();

const PROTOCOLS = ["csma", "tsch"] as const;

const problems = ref<ProblemInfoDto[]>([]);
const repositories = ref<SourceRepositoryDto[]>([]);
const name = ref("");
const problemId = ref("");
const protocol = ref<(typeof PROTOCOLS)[number]>("csma");
const firmware = ref("");
const duration = ref(DEFAULT_DURATION);
const panelMode = ref<"panel" | "chromosome">("panel");
const panelSize = ref(DEFAULT_PANEL);
const chromosome = ref<Record<string, unknown> | null>(null);
const seedMode = ref<"generated" | "imported">("generated");
const seedCount = ref(DEFAULT_SEEDS);
const masterSeed = ref(Math.floor(Math.random() * 2 ** 31));
const importedSeeds = ref<number[]>([]);
const fileName = ref("");
const fileError = ref("");
const objectives = ref<ObjectiveItem[]>(defaultObjectives());
const confidencePct = ref(DEFAULT_TARGETS.confidence * 100);
const errorPct = ref(DEFAULT_TARGETS.relative_error * 100);
const reliability = ref(DEFAULT_TARGETS.reliability);
const controlReplicates = ref(DEFAULT_CONTROL_REPLICATES);
const estimate = ref<StudyEstimate | null>(null);
const submitting = ref(false);
const error = ref("");

const metricNames = defaultDataConversion().metrics.map((metric) => metric.name);
const seedCheck = computed(() => checkSeeds(importedSeeds.value));
const seeds = computed(() =>
  seedMode.value === "generated" ? seedCount.value : importedSeeds.value.length,
);
const chromosomes = computed(() => (panelMode.value === "panel" ? panelSize.value : 1));
const cost = computed(() => studyCost(seeds.value, chromosomes.value, controlReplicates.value));
const hours = computed(() =>
  estimateHours(cost.value.total, estimate.value?.mean_seconds, props.pool),
);
const seedsValid = computed(() =>
  seedMode.value === "generated"
    ? seedCount.value >= MIN_SEEDS && seedCount.value <= MAX_SEEDS
    : importedSeeds.value.length > 0 && seedCheck.value.errors.length === 0,
);
const canSubmit = computed(
  () =>
    !submitting.value &&
    name.value.length > 0 &&
    problemId.value.length > 0 &&
    firmware.value.length > 0 &&
    seedsValid.value &&
    objectives.value.every((objective) => isValidSeedAggregator(objective.aggregator)) &&
    !cost.value.overLimit &&
    chromosomes.value >= 1 &&
    chromosomes.value <= MAX_CHROMOSOMES &&
    duration.value >= DURATION_RANGE[0] &&
    duration.value <= DURATION_RANGE[1],
);

function setAggregator(index: number, kind: string) {
  const objective = objectives.value[index]!;
  objective.aggregator = buildSeedAggregator(
    kind as SeedAggregatorKind,
    aggregatorTrim(objective.aggregator),
  );
}

function setTrim(index: number, raw: string) {
  objectives.value[index]!.aggregator = buildSeedAggregator("trimmed_mean", parseFloat(raw));
}

function addObjective() {
  objectives.value.push({ metric_name: metricNames[0]!, goal: "min", aggregator: "mean" });
}

async function onProblemChange() {
  chromosome.value = null;
  panelMode.value = "panel";
  try {
    const problem = await getProblem(problemId.value);
    if (!name.value) name.value = `seed study · ${problem.name}`;
    if (problem.draft.chromosome) {
      chromosome.value = exportChromosome(problem.draft.chromosome).chromosome as Record<
        string,
        unknown
      >;
    }
  } catch (e) {
    error.value = describe(e);
  }
}

function onSeedFile(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  importedSeeds.value = [];
  fileError.value = "";
  if (!file) return;
  fileName.value = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    const parsed = parseSeedFile(String(reader.result));
    if (parsed.ok) importedSeeds.value = parsed.seeds;
    else fileError.value = parsed.error;
  };
  reader.readAsText(file);
}

async function submit() {
  if (!canSubmit.value) return;
  submitting.value = true;
  error.value = "";
  try {
    // Exported the same way the experiment wizard does, so the study runs
    // exactly the configuration an optimisation of this problem would.
    const problem = await getProblem(problemId.value);
    const exported = exportProblem(problem.draft);
    const created = await createStudy({
      name: name.value,
      problem: exported.problem as unknown as Record<string, unknown>,
      source_repository_options: { [protocol.value]: firmware.value },
      duration: duration.value,
      seeds:
        seedMode.value === "generated"
          ? { mode: "generated", count: seedCount.value, master_seed: masterSeed.value }
          : { mode: "imported", values: importedSeeds.value, file_name: fileName.value },
      panel:
        panelMode.value === "panel"
          ? { source: "generated", size: panelSize.value }
          : { source: "provided" },
      chromosomes: panelMode.value === "chromosome" ? [chromosome.value!] : null,
      objectives: objectives.value,
      targets: {
        confidence: confidencePct.value / 100,
        relative_error: errorPct.value / 100,
        reliability: reliability.value,
      },
      control_replicates: controlReplicates.value,
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

async function loadEstimate() {
  if (duration.value < DURATION_RANGE[0] || duration.value > DURATION_RANGE[1]) {
    estimate.value = null;
    return;
  }
  try {
    estimate.value = await getStudyEstimate(duration.value);
  } catch {
    // The count is exact without it; only the time goes away.
    estimate.value = null;
  }
}

watch(duration, loadEstimate);

onMounted(async () => {
  try {
    const [p, r] = await Promise.all([listProblems(), getAllRepositories()]);
    problems.value = p;
    repositories.value = r;
    const named = r.find((repository) => repository.name.includes(protocol.value));
    firmware.value = (named ?? (r.length === 1 ? r[0] : undefined))?.id ?? "";
  } catch (e) {
    error.value = describe(e);
  }
  await loadEstimate();
});

watch(protocol, (next) => {
  const named = repositories.value.find((repository) => repository.name.includes(next));
  if (named) firmware.value = named.id;
});
</script>

<style scoped>
.study-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.study-form__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
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

.radio {
  display: flex;
  align-items: center;
  gap: 8px;
}

.radio > input {
  flex: none;
}

.inline-number {
  width: 72px;
  margin: 0 4px;
  padding: 4px 6px !important;
}

.inline-number--wide {
  width: 130px;
}

.seed-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.file-input {
  border: none !important;
  padding: 0 0 0 6px !important;
  background: transparent !important;
}

.objective-row {
  display: grid;
  grid-template-columns: minmax(150px, 1.4fr) 80px minmax(120px, 1fr) auto;
  gap: 8px;
  align-items: center;
}

.trim {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.trim input {
  width: 90px;
}

.target-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}

.target-row label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.hint {
  font-size: 11px;
  color: var(--color-text-muted);
  line-height: 1.45;
}

.hint--warn {
  color: #b45309;
}

.hint--error {
  color: var(--status-error);
}

.cost {
  margin: 0;
  font-size: 13px;
}

.cost__warn {
  color: var(--status-error);
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

@media (max-width: 640px) {
  .objective-row {
    grid-template-columns: 1fr 1fr;
  }
}
</style>
