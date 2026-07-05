<template>
  <div ref="containerEl" class="surface-wrap" />
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import 'echarts-gl'
import { useTheme } from '../../composables/useTheme'

const { isDark } = useTheme()

const containerEl = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null
let ro: ResizeObserver | null = null

// DTLZ2 M=3 Pareto front is the unit quarter-sphere in the positive orthant:
//   f1 = cos(α)·cos(β)
//   f2 = cos(α)·sin(β)
//   f3 = sin(α)
// with α,β ∈ [0, π/2].
// We pass these as parametric equations directly to echarts-gl.

function buildOption() {
  if (!chart) return
  const dark = isDark.value

  const surfaceColor  = dark ? '#f59e0b' : '#d97706'
  const wireColor     = dark ? 'rgba(245,158,11,0.18)' : 'rgba(180,100,0,0.15)'
  const axisColor     = dark ? '#45475a' : '#c0cad8'
  const labelColor    = dark ? '#7f849c' : '#64748b'
  const nameColor     = dark ? '#a6adc8' : '#374151'
  const splitColor    = dark ? '#313244' : '#dde3eb'
  const bgColor       = dark ? '#181825' : '#ffffff'
  const envColor      = dark ? '#181825' : '#ffffff'

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const axisConfig = (name: string): any => ({
    name,
    nameTextStyle: { color: nameColor, fontSize: 11, fontWeight: 700 },
    type: 'value',
    min: 0, max: 1,
    axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
    axisTick: { lineStyle: { color: axisColor } },
    axisLabel: { color: labelColor, fontSize: 10,
      formatter: (v: number) => v === 0 ? '0' : v === 1 ? '1' : '' },
    splitLine: { lineStyle: { color: splitColor, opacity: 0.5, width: 1 } },
  })

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const option: any = {
    backgroundColor: bgColor,
    grid3D: {
      show: true,
      boxWidth: 100, boxHeight: 100, boxDepth: 100,
      axisPointer: { show: false },
      splitLine: { show: true, lineStyle: { color: splitColor, opacity: 0.4 } },
      splitArea: { show: false },
      light: {
        main: {
          intensity: dark ? 0.8 : 1.1,
          shadow: false,
        },
        ambient: { intensity: dark ? 0.5 : 0.7 },
      },
      environment: envColor,
      viewControl: {
        autoRotate: true,
        autoRotateSpeed: 8,
        distance: 210,
        alpha: 18,
        beta: 40,
        zoomSensitivity: 1.2,
        rotateSensitivity: 1.2,
      },
      postEffect: {
        enable: dark,
        SSAO: { enable: dark, radius: 2, intensity: dark ? 0.8 : 0 },
      },
    },
    xAxis3D: axisConfig('f₁'),
    yAxis3D: axisConfig('f₂'),
    zAxis3D: axisConfig('f₃'),
    series: [
      {
        type: 'surface',
        coordinateSystem: 'cartesian3D',
        parametric: true,
        parametricEquation: {
          // α ∈ [0, π/2], β ∈ [0, π/2]
          u: { min: 0, max: Math.PI / 2, step: Math.PI / 2 / 40 },
          v: { min: 0, max: Math.PI / 2, step: Math.PI / 2 / 40 },
          x: (u: number, v: number) => Math.cos(u) * Math.cos(v),
          y: (u: number, v: number) => Math.cos(u) * Math.sin(v),
          z: (u: number)            => Math.sin(u),
        },
        itemStyle: {
          color: surfaceColor,
          opacity: dark ? 0.75 : 0.68,
        },
        wireframe: {
          show: true,
          lineStyle: { color: wireColor, width: 1 },
        },
        silent: true,
      },
    ],
  }

  chart.setOption(option, true)
}

onMounted(() => {
  if (!containerEl.value) return
  chart = echarts.init(containerEl.value)
  ro = new ResizeObserver(() => chart?.resize())
  ro.observe(containerEl.value)
  buildOption()
})

onBeforeUnmount(() => {
  ro?.disconnect()
  ro = null
  chart?.dispose()
  chart = null
})

watch(isDark, () => buildOption())
</script>

<style scoped>
.surface-wrap {
  width: 100%;
  height: 100%;
  min-height: 280px;
}
</style>
