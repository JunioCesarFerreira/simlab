<template>
  <div class="preview-panel">
    <div class="preview-header">
      <span class="preview-title">Theoretical Pareto front</span>
      <span class="preview-sub">{{ draft.benchmark }} · M={{ draft.M }}</span>
    </div>

    <div class="chart-area">
      <svg
        v-if="draft.M === 2"
        class="pareto-svg"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        <!-- Axes -->
        <line x1="20" y1="180" x2="190" y2="180" stroke="var(--color-border)" stroke-width="1"/>
        <line x1="20" y1="180" x2="20"  y2="10"  stroke="var(--color-border)" stroke-width="1"/>
        <!-- Axis labels -->
        <text x="190" y="188" font-size="9" fill="var(--color-text-muted)" text-anchor="end">f₁</text>
        <text x="14"  y="10"  font-size="9" fill="var(--color-text-muted)" text-anchor="middle">f₂</text>

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
        <circle cx="20" cy="10" r="3" fill="#10b981" opacity="0.6"/>
        <text x="24" y="10" font-size="8" fill="#10b981" dominant-baseline="middle">ideal</text>
      </svg>

      <!-- DTLZ2 M=3: interactive 3D quarter-sphere surface -->
      <div v-else-if="draft.benchmark === 'DTLZ2' && draft.M === 3" class="surface-3d-wrap">
        <Dtlz2Surface3D />
      </div>

      <!-- M≥3 fallback for ZDT1/SCH1 (always M=2) or DTLZ2 with M>3 -->
      <div v-else class="m3-placeholder">
        <div class="m3-icon">⬡</div>
        <p class="m3-text">
          {{ draft.benchmark }} with M={{ draft.M }}<br/>
          Pareto front: {{ frontDescription }}
        </p>
        <p class="m3-note">
          3D preview available in the experiment results<br/>
          after the run completes.
        </p>
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

// --- 2-D SVG sample points (M=2 only) ---
// Coordinate mapping: f ∈ [0,1] → SVG [20,190] (x) and [10,180] (y, inverted)
function mapX(f: number) { return 20 + f * 170 }
function mapY(f: number) { return 180 - f * 170 }

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
  min-height: 200px;
}
.pareto-svg {
  width: 100%;
  max-width: 280px;
  height: auto;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.06));
}

/* DTLZ2 M=3 surface */
.surface-3d-wrap {
  width: 100%;
  height: 300px;
  flex-shrink: 0;
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
