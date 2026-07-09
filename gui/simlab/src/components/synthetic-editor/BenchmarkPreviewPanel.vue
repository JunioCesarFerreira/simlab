<template>
  <div class="preview-panel">
    <div class="preview-header">
      <span class="preview-title">Theoretical Pareto front</span>
      <span class="preview-sub">{{ draft.benchmark }} · M={{ draft.M }}</span>
    </div>

    <div class="chart-area">
      <!-- M=2: SVG + formula side by side -->
      <div v-if="draft.M === 2" class="chart-row">
        <svg
          class="pareto-svg"
          viewBox="0 0 200 200"
          xmlns="http://www.w3.org/2000/svg"
        >
          <!-- Axes -->
          <line x1="38" y1="176" x2="192" y2="176" stroke="var(--color-border)" stroke-width="1"/>
          <line x1="38" y1="176" x2="38"  y2="8"   stroke="var(--color-border)" stroke-width="1"/>

          <!-- X axis ticks & labels -->
          <template v-for="v in [0, 0.25, 0.5, 0.75, 1]" :key="'xt'+v">
            <line :x1="mapX(v)" y1="176" :x2="mapX(v)" y2="180" stroke="var(--color-border)" stroke-width="0.8"/>
            <text :x="mapX(v)" y="188" font-size="7.5" fill="var(--color-text-muted)" text-anchor="middle">{{ v }}</text>
          </template>

          <!-- Y axis ticks & labels -->
          <template v-for="v in [0, 0.25, 0.5, 0.75, 1]" :key="'yt'+v">
            <line x1="38" :y1="mapY(v)" x2="34" :y2="mapY(v)" stroke="var(--color-border)" stroke-width="0.8"/>
            <text x="32" :y="mapY(v)" font-size="7.5" fill="var(--color-text-muted)" text-anchor="end" dominant-baseline="middle">{{ v }}</text>
          </template>

          <!-- Axis titles -->
          <text x="192" y="181" font-size="9" fill="var(--color-text-muted)" text-anchor="end">f₁</text>
          <text x="32"  y="7"   font-size="9" fill="var(--color-text-muted)" text-anchor="middle">f₂</text>

          <!-- Pareto front curve -->
          <polyline
            :points="svgPoints"
            fill="none"
            stroke="#f59e0b"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />

          <!-- Ideal point -->
          <circle :cx="mapX(0)" :cy="mapY(1)" r="3" fill="#10b981" opacity="0.6"/>
          <text :x="mapX(0)+5" :y="mapY(1)" font-size="8" fill="#10b981" dominant-baseline="middle">ideal</text>
        </svg>

        <!-- Formula panel -->
        <div class="formula-panel">
          <div class="formula-title">{{ formulaData.title }}</div>
          <div class="formula-body">
            <div v-for="(line, i) in formulaData.lines" :key="i" class="formula-line" v-html="line" />
            <div v-if="formulaData.gDef" class="formula-gdef" v-html="formulaData.gDef" />
            <div class="formula-domain" v-html="formulaData.domain" />
          </div>
        </div>
      </div>

      <!-- DTLZ2 M=3: interactive 3D quarter-sphere surface + formula -->
      <div v-else-if="draft.benchmark === 'DTLZ2' && draft.M === 3" class="chart-row">
        <div class="surface-3d-wrap">
          <Dtlz2Surface3D />
        </div>
        <div class="formula-panel">
          <div class="formula-title">{{ formulaData.title }}</div>
          <div class="formula-body">
            <div v-for="(line, i) in formulaData.lines" :key="i" class="formula-line" v-html="line" />
            <div v-if="formulaData.gDef" class="formula-gdef" v-html="formulaData.gDef" />
            <div class="formula-domain" v-html="formulaData.domain" />
          </div>
        </div>
      </div>

      <!-- DTLZ2 M>3: text placeholder + formula -->
      <div v-else-if="draft.benchmark === 'DTLZ2'" class="chart-row chart-row--placeholder">
        <div class="m3-placeholder">
          <div class="m3-icon">⬡</div>
          <p class="m3-text">
            DTLZ2 with M={{ draft.M }}<br/>
            {{ frontDescription }}
          </p>
          <p class="m3-note">3D preview available in experiment results after the run completes.</p>
        </div>
        <div class="formula-panel">
          <div class="formula-title">{{ formulaData.title }}</div>
          <div class="formula-body">
            <div v-for="(line, i) in formulaData.lines" :key="i" class="formula-line formula-line--sm" v-html="line" />
            <div v-if="formulaData.gDef" class="formula-gdef formula-gdef--sm" v-html="formulaData.gDef" />
            <div class="formula-domain" v-html="formulaData.domain" />
          </div>
        </div>
      </div>

      <!-- Other fallback (ZDT1/SCH1 always M=2, shouldn't reach here) -->
      <div v-else class="m3-placeholder">
        <div class="m3-icon">⬡</div>
        <p class="m3-text">{{ draft.benchmark }} — M={{ draft.M }}</p>
      </div>
    </div>

    <!-- Info cards -->
    <div class="info-section">
      <div class="info-card">
        <span class="info-key">Front shape</span>
        <span class="info-val">{{ frontDescription }}</span>
      </div>
      <div class="info-card">
        <span class="info-key">Decision vars</span>
        <span class="info-val">{{ draft.nVars }} (genome: {{ draft.nVars }} floats)</span>
      </div>
      <div class="info-card">
        <span class="info-key">Domain</span>
        <span class="info-val">x ∈ [0,1]ⁿ (evaluated directly)</span>
      </div>
      <div v-if="draft.noiseStd > 0" class="info-card info-card--warn">
        <span class="info-key">Noise</span>
        <span class="info-val">σ = {{ draft.noiseStd }} (stochastic)</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent } from 'vue'
import { useSyntheticStore } from '../../app/stores/syntheticStore'

const Dtlz2Surface3D = defineAsyncComponent(
  () => import('./Dtlz2Surface3D.vue')
)

const store = useSyntheticStore()
const draft = computed(() => store.draft)

// --- Theoretical front descriptions ---
const frontDescription = computed(() => {
  switch (draft.value.benchmark) {
    case 'ZDT1': return 'Convex curve  f₂ = 1 − √f₁'
    case 'SCH1':  return 'Two parabola arcs  f₁=x², f₂=(x−2)²'
    default:      return `Unit hypersphere segment in Q₁ (M=${draft.value.M})`
  }
})

// --- Formula panels ---
interface FormulaInfo { title: string; lines: string[]; gDef?: string; domain: string }

const dot = '&thinsp;&middot;&thinsp;'
const cx  = (i: number) => `cos(&pi;x<sub>${i}</sub>/2)`
const sx  = (i: number) => `sin(&pi;x<sub>${i}</sub>/2)`

// One explicit DTLZ2 objective line in terms of decision variables x
function dtlz2ObjLine(j: number, M: number): string {
  const parts: string[] = []
  if (j === 1) {
    for (let i = 1; i <= M - 1; i++) parts.push(cx(i))
  } else {
    for (let i = 1; i <= M - j; i++) parts.push(cx(i))
    parts.push(sx(M - j + 1))
  }
  parts.push('(1 + g)')
  return `f<sub>${j}</sub>(x) = ${parts.join(dot)}`
}

function dtlz2Formula(M: number, n: number): FormulaInfo {
  const gDef = `g(x) = &sum;<sub>i=${M}</sub><sup>n</sup>(x<sub>i</sub> &minus; 0.5)<sup>2</sup>`
  const domain = `x &isin; [0,1]<sup>n</sup>,&ensp;n = ${n},&ensp;M = ${M}`

  if (M > 5) {
    return {
      title: `Objectives (M = ${M})`,
      lines: [
        `f<sub>k</sub>(x) = (&prod;<sub>i=1</sub><sup>M&minus;k</sup>${cx(0).replace('x<sub>0</sub>', 'x<sub>i</sub>')})${dot}${sx(0).replace('x<sub>0</sub>', 'x<sub>M&minus;k+1</sub>')}${dot}(1+g)`,
        `f<sub>${M}</sub>(x) = ${sx(1)}${dot}(1 + g)`,
      ],
      gDef,
      domain,
    }
  }

  return {
    title: 'Objective functions',
    lines: Array.from({ length: M }, (_, i) => dtlz2ObjLine(i + 1, M)),
    gDef,
    domain,
  }
}

const formulaData = computed((): FormulaInfo => {
  const { benchmark, M } = draft.value
  const n = draft.value.nVars   // genome size = number of decision variables

  switch (benchmark) {
    case 'ZDT1': return {
      title: 'Objective functions',
      lines: [
        'f<sub>1</sub>(x) = x<sub>1</sub>',
        'f<sub>2</sub>(x) = g(x) &thinsp;&middot;&thinsp; (1 &minus; &radic;(f<sub>1</sub>/g))',
      ],
      gDef: `g(x) = 1 + 9/(n&minus;1) &thinsp;&middot;&thinsp; &sum;<sub>i=2</sub><sup>n</sup>x<sub>i</sub>`,
      domain: `x &isin; [0,1]<sup>n</sup>,&ensp;n = ${n}`,
    }
    case 'SCH1': return {
      title: 'Objective functions',
      lines: [
        'f<sub>1</sub>(x) = x<sup>2</sup>',
        'f<sub>2</sub>(x) = (x &minus; 2)<sup>2</sup>',
      ],
      domain: 'x &isin; [0, 2]',
    }
    default: return dtlz2Formula(M, n)
  }
})

// --- 2-D SVG sample points (M=2 only) ---
// Coordinate mapping: f ∈ [0,1] → SVG axes
function mapX(f: number) { return 38 + f * 154 }
function mapY(f: number) { return 176 - f * 168 }

const svgPoints = computed((): string => {
  const N = 80
  const pts: string[] = []

  if (draft.value.benchmark === 'ZDT1') {
    for (let i = 0; i <= N; i++) {
      const f1 = i / N
      const f2 = 1 - Math.sqrt(f1)
      pts.push(`${mapX(f1)},${mapY(f2)}`)
    }
  } else if (draft.value.benchmark === 'SCH1') {
    // f1=x², f2=(x-2)², x∈[0,2] → normalised by max(f1)=4, max(f2)=4
    for (let i = 0; i <= N; i++) {
      const x = (i / N) * 2
      const f1 = Math.min(1, x * x / 4)
      const f2 = Math.min(1, (x - 2) * (x - 2) / 4)
      pts.push(`${mapX(f1)},${mapY(f2)}`)
    }
  } else {
    // DTLZ2 M=2: unit circle arc θ ∈ [0, π/2]
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * Math.PI / 2
      const f1 = Math.cos(t)
      const f2 = Math.sin(t)
      pts.push(`${mapX(f1)},${mapY(f2)}`)
    }
  }
  return pts.join(' ')
})
</script>

<style scoped>
.preview-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  height: 100%;
  overflow-y: auto;
}
.preview-header {
  display: flex;
  align-items: baseline;
  gap: 10px;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 10px;
}
.preview-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-muted);
}
.preview-sub { font-size: 12px; color: var(--color-text); font-weight: 600; }

.chart-area {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
}

/* M=2: SVG + formula side-by-side */
.chart-row {
  display: flex;
  align-items: center;
  gap: 20px;
  width: 100%;
  height: 100%;
}
.pareto-svg {
  /* fill chart-area height, maintain square aspect ratio, never overflow */
  height: 100%;
  width: auto;
  max-height: 100%;
  max-width: 62%;
  flex-shrink: 0;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.06));
}

.formula-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px 16px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  min-width: 140px;
}
.formula-title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-muted);
}
.formula-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.chart-row--placeholder {
  align-items: flex-start;
  padding-top: 16px;
}

.formula-line {
  font-size: 14px;
  font-family: "Georgia", "Times New Roman", serif;
  color: var(--color-text);
  line-height: 1.6;
  white-space: nowrap;
}
.formula-line--sm {
  font-size: 12px;
}
.formula-gdef {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--color-border);
  font-size: 12px;
  font-family: "Georgia", "Times New Roman", serif;
  color: var(--color-text-muted);
  white-space: nowrap;
}
.formula-gdef--sm { font-size: 11px; }

.formula-domain {
  margin-top: 4px;
  font-size: 11px;
  font-family: "Georgia", "Times New Roman", serif;
  color: var(--color-text-muted);
  border-top: 1px solid var(--color-border);
  padding-top: 6px;
}

/* DTLZ2 M=3 surface — inside chart-row it takes remaining horizontal space */
.surface-3d-wrap {
  flex: 1;
  height: 100%;
  min-height: 300px;
}

/* M≥3 placeholder */
.m3-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 24px;
  text-align: center;
}
.m3-icon { font-size: 36px; color: #f59e0b; opacity: 0.7; }
.m3-text { font-size: 13px; font-weight: 600; color: var(--color-text); line-height: 1.5; }
.m3-note { font-size: 11px; color: var(--color-text-muted); line-height: 1.5; }

/* Info cards */
.info-section { display: flex; flex-direction: column; gap: 6px; }
.info-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 7px 12px;
  font-size: 11px;
}
.info-card--warn {
  border-color: rgba(245, 158, 11, 0.4);
  background: rgba(245, 158, 11, 0.06);
}
.info-key { color: var(--color-text-muted); }
.info-val { font-weight: 600; color: var(--color-text); font-family: monospace; }
</style>
