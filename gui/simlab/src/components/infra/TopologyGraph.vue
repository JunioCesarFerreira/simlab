<template>
  <div class="topo-graph">
    <div
      ref="el"
      class="topo-graph__canvas"
      :style="{ height: `${height}px` }"
      role="img"
      :aria-label="ariaLabel"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { EChartsOption, GraphSeriesOption } from "echarts";
import { useEChart } from "../../composables/useEChart";
import { useTheme } from "../../composables/useTheme";
import { chartPalette } from "../../services/chartTheme";
import {
  MASTER_NODE_ID,
  experimentColors,
  hostLoad,
  hostNodeId,
  hostTone,
  hostTooltip,
  layoutTopology,
  masterTone,
  masterTooltip,
  summarize,
  toneColors,
  workerNodeId,
  workerTone,
  workerTooltip,
  type Tone,
} from "../../services/topology";
import type { TopologyDto } from "../../types/infra";

/**
 * master-node → hosts → workers, coloured by status, redrawn on every poll.
 *
 * Fixed positions rather than a force layout: the picture is refreshed every
 * few seconds while experiments run, and nodes that move on every refresh
 * are impossible to follow. Only colours, widths and labels change.
 */
const props = defineProps<{
  topology: TopologyDto;
  /** Worker names under each node — off on narrow screens, where they
   *  collide; the tooltip still carries them. */
  showWorkerLabels?: boolean;
}>();

const el = ref<HTMLElement | null>(null);
const chart = useEChart(el);
const { isDark } = useTheme();

const layout = computed(() => layoutTopology(props.topology));
/** One host line is this tall. Room for a host box and a worker label. */
const ROW_PX = 58;
const height = computed(() => layout.value.rows * ROW_PX + 48);

const ariaLabel = computed(() => {
  const s = summarize(props.topology);
  return (
    `Infrastructure graph: master-node ${props.topology.master.status}, ` +
    `${s.hostsReachable} of ${s.hosts} hosts reachable, ${s.running} of ${s.workers} ` +
    `workers running a simulation, ${s.free} free.`
  );
});

// Custom per-node payload: the tooltip text travels with the node, so the
// formatter does not have to look anything up.
type NodeDatum = NonNullable<GraphSeriesOption["data"]>[number] & { tip?: string };

function option(topology: TopologyDto, dark: boolean, showLabels: boolean): EChartsOption {
  const c = chartPalette(dark);
  const tone = toneColors(dark);
  const colorOf = (t: Tone) => tone[t];
  const exp = experimentColors(topology.running_experiments);
  const positions = new Map(layout.value.nodes.map((n) => [n.id, n]));

  const nodes: NodeDatum[] = [];
  const links: NonNullable<GraphSeriesOption["links"]> = [];

  const masterPos = positions.get(MASTER_NODE_ID)!;
  const mTone = masterTone(topology.master.status, topology.master.dns);
  nodes.push({
    id: MASTER_NODE_ID,
    name: "master-node",
    value: [masterPos.x, masterPos.y],
    symbol: "roundRect",
    symbolSize: [110, 34],
    itemStyle: { color: c.tooltip, borderColor: colorOf(mTone), borderWidth: 2.5 },
    label: { show: true, color: c.text, fontSize: 12, fontWeight: 600 },
    tip: masterTooltip(topology.master),
  });

  for (const host of topology.hosts) {
    const pos = positions.get(hostNodeId(host.id))!;
    const hTone = hostTone(host);
    const load = hostLoad(host);
    nodes.push({
      id: hostNodeId(host.id),
      name: `${host.name}\n${load.running}/${load.total} running`,
      value: [pos.x, pos.y],
      symbol: "roundRect",
      symbolSize: [108, 40],
      itemStyle: { color: c.tooltip, borderColor: colorOf(hTone), borderWidth: 2 },
      label: { show: true, color: c.text, fontSize: 11, lineHeight: 15 },
      tip: hostTooltip(host, topology.generated_at),
    });
    links.push({
      source: MASTER_NODE_ID,
      target: hostNodeId(host.id),
      lineStyle: {
        color: colorOf(hTone),
        width: 2,
        type: hTone === "bad" || hTone === "off" ? "dashed" : "solid",
        opacity: 0.9,
      },
    });

    for (const worker of host.workers) {
      const wPos = positions.get(workerNodeId(worker.id))!;
      const wTone = workerTone(worker);
      const accent = worker.run?.experiment_id ? exp[worker.run.experiment_id] : undefined;
      nodes.push({
        id: workerNodeId(worker.id),
        name: worker.name,
        value: [wPos.x, wPos.y],
        symbol: "circle",
        symbolSize: wTone === "busy" ? 20 : 15,
        itemStyle: {
          color: colorOf(wTone),
          // The experiment it runs, as a ring — two experiments sharing the
          // pool are told apart without a second legend.
          borderColor: accent ?? c.tooltip,
          borderWidth: accent ? 3.5 : 1.5,
        },
        label: {
          show: showLabels,
          position: "bottom",
          distance: 4,
          color: c.muted,
          fontSize: 10,
        },
        tip: workerTooltip(worker, topology.generated_at),
      });
      links.push({
        source: hostNodeId(host.id),
        target: workerNodeId(worker.id),
        lineStyle: {
          color: colorOf(wTone),
          // Straight, every edge of a host lies on the same horizontal line
          // and the workers read as a daisy chain — cooja1 to cooja2 to
          // cooja3 — when each is linked to the host directly. A small,
          // constant curveness bows each edge in proportion to its length, so
          // they nest as a fan with one arc landing on each worker.
          curveness: 0.08,
          width: wTone === "busy" ? 2.5 : 1.2,
          type: wTone === "bad" || wTone === "off" ? "dashed" : "solid",
          opacity: wTone === "off" ? 0.5 : 0.85,
        },
      });
    }
  }

  const b = layout.value.bounds;
  return {
    backgroundColor: c.bg,
    animationDurationUpdate: 300,
    // Hidden axes, not a fitted view: `layout: "none"` scales x and y
    // independently to fill the box and stretches the symbols with them —
    // circles drawn as ellipses, boxes twice as tall as specified. On a
    // cartesian grid the axes place the points and the symbols stay in pixels.
    grid: { left: 64, right: 28, top: 22, bottom: 22 },
    xAxis: { type: "value", min: b.xMin, max: b.xMax, show: false },
    yAxis: { type: "value", min: b.yMin, max: b.yMax, inverse: true, show: false },
    tooltip: {
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params: unknown) => {
        const p = params as { dataType?: string; data?: { tip?: string } };
        return p.dataType === "edge" ? "" : (p.data?.tip ?? "");
      },
    },
    series: [
      {
        type: "graph",
        coordinateSystem: "cartesian2d",
        roam: false,
        edgeSymbol: ["none", "none"],
        emphasis: { focus: "adjacency", lineStyle: { width: 3 } },
        data: nodes,
        links,
      },
    ],
  };
}

function render() {
  chart.setOption(option(props.topology, isDark.value, props.showWorkerLabels ?? true));
}

watch(() => [props.topology, props.showWorkerLabels], render, { deep: true, immediate: true });
watch(isDark, render);
watch(chart.ready, (ready) => ready && render());
</script>

<style scoped>
/* On a phone the graph keeps a readable width and scrolls inside its card,
   as the tables do, rather than squeezing ten hosts into 390 px. */
.topo-graph {
  overflow-x: auto;
}

.topo-graph__canvas {
  width: 100%;
  min-width: 660px;
}
</style>
