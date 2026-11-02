/**
 * How each problem's solution is read into a drawing.
 */
import { describe, expect, it } from "vitest";
import { chartPalette } from "../chartTheme";
import {
  equalScale,
  samplePath,
  solutionScene,
  topologyOption,
  type NetworkScene,
} from "../solutionTopology";

const REGION = [-150, -150, 150, 150];
const MOBILE = {
  name: "node13",
  is_closed: true,
  path_segments: [["70", "44 + 31 * t"], ["70 - 140 * t", "75"]],
};

function network(scene: ReturnType<typeof solutionScene>): NetworkScene {
  expect(scene.kind).toBe("network");
  return scene as NetworkScene;
}

function labels(scene: ReturnType<typeof solutionScene>): string[] {
  return scene.facts.map((f) => f.label);
}

describe("P1, relays placed freely", () => {
  const problem = { name: "problem1", region: REGION, sink: [0, 0], radius_of_reach: 60, mobile_nodes: [MOBILE] };

  it("draws the relays, the sink and the links within reach", () => {
    const scene = network(solutionScene(problem, {
      mac_protocol: 0,
      relays: [{ x: 50, y: 0 }, { x: 100, y: 0 }, { x: -140, y: 140 }],
    }));

    expect(scene.motes).toEqual([[50, 0], [100, 0], [-140, 140]]);
    expect(scene.base).toEqual({ at: [0, 0], label: "Sink" });
    // sink–50 and 50–100 are within 60; the corner relay is alone.
    expect(scene.links).toHaveLength(2);
    expect(scene.region).toEqual(REGION);
    expect(scene.disks).toEqual({ radius: 60, label: "Radius of reach", aroundBase: true });
  });

  it("counts the relays with a multi-hop path to the sink", () => {
    const scene = solutionScene(problem, {
      mac_protocol: 1,
      relays: [{ x: 50, y: 0 }, { x: 100, y: 0 }, { x: -140, y: 140 }],
    });

    expect(labels(scene)).toEqual(["3 relays", "MAC tsch", "2/3 reach the sink", "1 mobile nodes"]);
    expect(scene.facts[2]?.tone).toBe("warn");
  });

  it("samples the trajectories of the mobile nodes", () => {
    const scene = network(solutionScene(problem, { mac_protocol: 0, relays: [] }));

    const path = scene.paths[0]!.points;
    expect(path[0]).toEqual([70, 44]);
    expect(path[path.length - 1]).toEqual([70, 44]); // closed
  });
});

describe("P2, a selection over candidates", () => {
  const problem = {
    name: "problem2", region: REGION, sink: [0, 0], radius_of_reach: 60,
    candidates: [[50, 0], [100, 0], [0, 100]],
  };

  it("splits the candidates into selected and unused, in mask order", () => {
    const scene = network(solutionScene(problem, { mac_protocol: 0, mask: [1, 0, 1] }));

    expect(scene.motes).toEqual([[50, 0], [0, 100]]);
    expect(scene.unused).toEqual([[100, 0]]);
    expect(labels(scene)[0]).toBe("2 of 3 candidates");
    expect(scene.targets).toEqual([]);
  });
});

describe("P3, sensing coverage", () => {
  const problem = {
    name: "problem3", region: REGION, sink: [0, 0], radius_of_reach: 100,
    radius_of_cover: 30, k_required: 2,
    candidates: [[0, 50], [20, 50], [100, 100]],
    targets: [[10, 50], [100, 120]],
  };

  it("counts how many selected motes cover each target, against k", () => {
    const scene = network(solutionScene(problem, { mac_protocol: 0, mask: [1, 1, 1] }));

    expect(scene.targets.map((t) => t.covered)).toEqual([2, 1]);
    expect(scene.required).toBe(2);
    // The sink senses nothing, so the coverage radius is not drawn around it.
    expect(scene.disks).toEqual({ radius: 30, label: "Radius of coverage", aroundBase: false });
    expect(labels(scene)).toContain("1/2 targets covered 2×");
  });
});

describe("P4, a mobile sink's route", () => {
  const problem = {
    name: "problem4", region: REGION, sink_base: [0, 0],
    nodes: [[10, 10], [90, 10], [140, -140]],
    sojourns: [
      { id: 0, position: [0, 0], adjacency: [1, 2], visibleNodes: [0] },
      { id: 1, position: [80, 0], adjacency: [0, 2], visibleNodes: [1] },
      { id: 2, position: [-80, 0], adjacency: [0, 1], visibleNodes: [] },
    ],
  };

  it("follows the route in order and reads the stops it never makes", () => {
    const scene = network(solutionScene(problem, {
      mac_protocol: 0, route: [0, 1, 0], sojourn_times: [0, 12.5, 0],
    }));

    expect(scene.route.map((s) => s.location)).toEqual([0, 1, 0]);
    expect(scene.route[1]).toEqual({ at: [80, 0], dwell: 12.5, location: 1 });
    expect(scene.unused).toEqual([[-80, 0]]);
    expect(scene.given).toHaveLength(3);
    expect(scene.base?.label).toBe("Base");
    // collected: mote 0 from the base, mote 1 from L1; mote 2 from nowhere.
    expect(scene.links).toHaveLength(2);
    expect(labels(scene)).toEqual([
      "2 of 3 stops visited", "13 s of stops", "MAC csma", "2/3 motes in reach of a stop",
    ]);
  });
});

describe("P0 and the rest", () => {
  it("shows a benchmark's decision vector, since it has no network", () => {
    const scene = solutionScene({ name: "problem0", n: 2 }, { x: [0.25, 0.75] });

    expect(scene).toMatchObject({ kind: "vector", values: [0.25, 0.75] });
  });

  it("says so for a problem it does not know", () => {
    expect(solutionScene({ name: "problem9" }, {}).kind).toBe("none");
  });

  it("frames an instance without a region around what it draws", () => {
    const scene = network(solutionScene(
      { name: "problem1", sink: [0, 0], radius_of_reach: 10 },
      { relays: [{ x: 40, y: -20 }] },
    ));

    expect(scene.region).toEqual([0, -20, 40, 0]);
  });
});

describe("drawing", () => {
  const c = chartPalette(false);

  it("keeps one scale on both axes", () => {
    const { xMin, xMax, yMin, yMax } = equalScale([-100, -100, 100, 100], 880, 476);

    expect((xMax - xMin) / 800).toBeCloseTo((yMax - yMin) / 400, 6);
    expect(yMax - yMin).toBeGreaterThanOrEqual(200);
  });

  it("starts with the radius disks off, behind a legend entry", () => {
    const scene = solutionScene(
      { name: "problem1", region: REGION, sink: [0, 0], radius_of_reach: 60 },
      { relays: [{ x: 50, y: 0 }] },
    );

    const option = topologyOption(scene, c, 800, 480) as { legend: { selected: Record<string, boolean>; data: string[] } };

    expect(option.legend.selected).toEqual({ "Radius of reach": false });
    expect(option.legend.data).toEqual(["Radius of reach", "In reach", "Relays", "Sink"]);
  });

  it("drops a segment it cannot evaluate instead of failing", () => {
    expect(samplePath([["t", "oops("]], false, 4)).toEqual([]);
  });
});
