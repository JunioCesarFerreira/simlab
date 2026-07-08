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
            <div
              v-for="(line, i) in formulaData.lines"
              :key="i"
              class="formula-line"
              v-html="line"
            />
            <div v-if="formulaData.domain" class="formula-domain" v-html="formulaData.domain" />
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
            <div v-if="formulaData.domain" class="formula-domain" v-html="formulaData.domain" />
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
            <div v-if="formulaData.domain" class="formula-domain" v-html="formulaData.domain" />
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
        <span class="info-val">{{ draft.nVars }} (genome: {{ 2 * nRelays }} floats)</span>
      </div>
      <div class="info-card">
        <span class="info-key">Normalised to</span>
        <span class="info-val">[0,1]ⁿ via region Ω</span>
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
const nRelays = computed(() => store.nRelays())

// --- Theoretical front descriptions ---
const frontDescription = computed(() => {
  switch (draft.value.benchmark) {
    case 'ZDT1': return 'Convex curve  f₂ = 1 − √f₁'
    case 'SCH1':  return 'Two parabola arcs  f₁=x², f₂=(x−2)²'
    default:      return `Unit hypersphere segment in Q₁ (M=${draft.value.M})`
  }
})

// --- Formula panels ---
interface FormulaInfo { title: string; lines: string[]; domain?: string }

// Build angular parameter names for DTLZ2 (M-1 parameters)
function dtlz2Params(M: number): string[] {
  if (M === 2) return ['&theta;']
  if (M === 3) return ['&alpha;', '&beta;']
  return Array.from({ length: M - 1 }, (_, i) => `&theta;<sub>${i + 1}</sub>`)
}

// Build one explicit line: f_j(params) = cos θ₁·…·cos θ_{M-j}·sin θ_{M-j+1}
// For j=1 all cosines; for j=M just sin θ₁
function dtlz2Line(j: number, M: number, params: string[]): string {
  const sub = `<sub>${j}</sub>`
  const paramStr = params.join(', ')
  const dot = '&thinsp;&middot;&thinsp;'

  if (j === 1) {
    // product of all M-1 cosines
    return `f${sub}(${paramStr}) = ${params.map(p => `cos ${p}`).join(dot)}`
  }
  // M - j cosines followed by one sine
  const cosTerms = params.slice(0, M - j).map(p => `cos ${p}`).join(dot)
  const sinTerm  = `sin ${params[M - j]!}`
  return `f${sub}(${paramStr}) = ${cosTerms ? cosTerms + dot : ''}${sinTerm}`
}

function dtlz2Formula(M: number): FormulaInfo {
  const params  = dtlz2Params(M)
  const domain  = `${params.join(', ')} &isin; [0, &pi;/2]`

  if (M > 5) {
    // General notation for many objectives
    const n = M - 1
    return {
      title: `Parametric form (M = ${M})`,
      lines: [
        `f<sub>1</sub> = &prod;<sub>i=1</sub><sup>${n}</sup> cos &theta;<sub>i</sub>`,
        `f<sub>k</sub> = (&prod;<sub>i=1</sub><sup>M&minus;k</sup> cos &theta;<sub>i</sub>) sin &theta;<sub>M&minus;k+1</sub>`,
        `f<sub>${M}</sub> = sin &theta;<sub>1</sub>`,
      ],
      domain: `&theta;<sub>i</sub> &isin; [0, &pi;/2],  k = 2&hellip;${M - 1}`,
    }
  }

  return {
    title: 'Parametric form',
    lines: Array.from({ length: M }, (_, i) => dtlz2Line(i + 1, M, params)),
    domain,
  }
}

const formulaData = computed((): FormulaInfo => {
  const { benchmark, M } = draft.value
  switch (benchmark) {
    case 'ZDT1': return {
      title: 'Pareto front',
      lines: ['f<sub>2</sub> = 1 &minus; &radic;f<sub>1</sub>'],
      domain: 'f<sub>1</sub> &isin; [0, 1]',
    }
    case 'SCH1': return {
      title: 'Parametric form',
      lines: [
        'f<sub>1</sub>(x) = x<sup>2</sup>',
        'f<sub>2</sub>(x) = (x &minus; 2)<sup>2</sup>',
      ],
      domain: 'x &isin; [0, 2]',
    }
    default: return dtlz2Formula(M)   // DTLZ2 — any M
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
.formula-domain {
  margin-top: 4px;
  font-size: 12px;
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
