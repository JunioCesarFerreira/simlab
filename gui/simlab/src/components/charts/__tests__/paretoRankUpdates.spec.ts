// @vitest-environment vue-renderer
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRenderer, defineComponent, h, nextTick, shallowRef, type Component } from "vue";
import ParetoFrontChart from "../ParetoFrontChart.vue";
import ParetoFront3DChart from "../ParetoFront3DChart.vue";
import type { GenerationDto } from "../../../types/simlab";
import { computeRanksWithDuplicates } from "../../../utils/nonDominatedSort";

const charts = vi.hoisted(() => ({
  twoD: { setOption: vi.fn(), on: vi.fn(), dispatch: vi.fn(), exportImage: vi.fn() },
  threeD: { setOption: vi.fn(), on: vi.fn(), dispose: vi.fn(), resize: vi.fn() },
}));
vi.mock("../../../composables/useEChart", async () => {
  const { ref } = await import("vue");
  return { useEChart: () => ({ ...charts.twoD, ready: ref(true) }) };
});
vi.mock("../../../lib/echarts", () => ({ init: () => charts.threeD }));
vi.mock("echarts-gl", () => ({}));
vi.mock("../../../composables/useTheme", async () => {
  const { ref } = await import("vue");
  return { useTheme: () => ({ isDark: ref(false) }) };
});

// Mount the real Vue components with an in-memory host. Only ECharts/WebGL
// are mocked: their received options reveal whether delayed ranks repaint.
interface HostNode {
  parent: HostNode | null;
  children: HostNode[];
  text: string;
  props: Record<string, unknown>;
  options: HostNode[];
  addEventListener: () => void;
}
const node = (): HostNode => ({
  parent: null, children: [], text: "", props: {}, options: [], addEventListener() {},
});
function remove(child: HostNode) {
  const siblings = child.parent?.children;
  if (siblings) siblings.splice(siblings.indexOf(child), 1);
  child.parent = null;
}
const renderer = createRenderer<HostNode, HostNode>({
  createElement: node,
  createText: node,
  createComment: node,
  setText: (el, text) => { el.text = text; },
  setElementText: (el, text) => { el.text = text; },
  parentNode: (el) => el.parent,
  nextSibling: (el) => el.parent?.children[(el.parent.children.indexOf(el)) + 1] ?? null,
  patchProp: (el, key, _previous, value) => { el.props[key] = value; },
  remove,
  insert: (el, parent, anchor) => {
    if (el.parent) remove(el);
    el.parent = parent;
    parent.children.splice(anchor ? parent.children.indexOf(anchor) : parent.children.length, 0, el);
  },
});

const population = [
  { id: "early", individual_id: "early", objectives: [3, 3, 3], chromosome: { x: 0 } },
  { id: "middle", individual_id: "middle", objectives: [2, 2, 2], chromosome: { x: 1 } },
  { id: "late", individual_id: "late", objectives: [1, 1, 1], chromosome: { x: 2 } },
];
const generations: GenerationDto[] = population.map((individual, index) => ({
  id: `gen-${index}`, experiment_id: "experiment", index, status: "Done",
  population: [{ ...individual, topology_picture_id: null }],
}));
const props = {
  generations,
  objectiveNames: ["f1", "f2", "f3"],
  objectiveGoals: ["min", "min", "min"],
  paretoFront: [{ chromosome: { x: 2 }, objectives: { f1: 1, f2: 1, f3: 1 } }],
};
const unmounts: (() => void)[] = [];
beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
});
afterEach(() => {
  unmounts.splice(0).forEach((unmount) => unmount());
  vi.unstubAllGlobals();
});

describe.each([
  ["2D", ParetoFrontChart, charts.twoD],
  ["3D", ParetoFront3DChart, charts.threeD],
] as const)("%s Pareto colors", (_name, component, chart) => {
  it("repaints all generations when worker ranks arrive, with no pinned points", async () => {
    const ranks = shallowRef(new Map<string, number>());
    const app = renderer.createApp(defineComponent({
      setup: () => () => h(component as Component, { ...props, rankMap: ranks.value }),
    }));
    app.mount(node());
    unmounts.push(() => app.unmount());
    await nextTick();

    expect(chart.setOption).toHaveBeenCalled();
    expect(chart.setOption.mock.lastCall![0].series.map((s: { name: string }) => s.name))
      .toEqual(["Population", "Pareto Front"]);
    chart.setOption.mockClear();

    // Only rankMap changes, as happens when the background worker finishes.
    ranks.value = computeRanksWithDuplicates(population, [true, true, true]);
    await nextTick();

    expect(chart.setOption).toHaveBeenCalledOnce();
    const [option, updateOptions] = chart.setOption.mock.lastCall!;
    expect(option.series.map((s: { name: string; itemStyle: { color: string }; data: { individualId: string }[] }) => ({
      name: s.name, color: s.itemStyle.color, ids: s.data.map((p) => p.individualId),
    }))).toEqual([
      { name: "Front 1", color: "#3b82f6", ids: ["late"] },
      { name: "Front 2", color: "#10b981", ids: ["middle"] },
      { name: "Front 3", color: "#f59e0b", ids: ["early"] },
    ]);
    // Recoloring must preserve the user's zoom/camera state.
    expect(updateOptions).toEqual({ replaceMerge: ["series"] });
  });
});
