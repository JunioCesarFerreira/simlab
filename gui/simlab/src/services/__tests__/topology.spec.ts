/**
 * The topology page's readings: colours, positions, tooltips.
 *
 * The tooltips carry names and messages written by operators and returned by
 * remote machines, and ECharts renders them as HTML — so escaping is pinned
 * here, next to everything else the page says.
 */
import { describe, expect, it } from "vitest";
import {
  EXPERIMENT_COLORS,
  MASTER_NODE_ID,
  dnsLines,
  elapsedSeconds,
  escapeHtml,
  experimentColors,
  formatElapsed,
  hostLoad,
  hostNodeId,
  hostTone,
  hostTooltip,
  layoutTopology,
  masterTone,
  masterTooltip,
  summarize,
  workerLabel,
  workerNodeId,
  workerTone,
  workerTooltip,
  WORKERS_PER_ROW,
} from "../topology";
import type { MasterDns, TopologyHost, TopologyWorker } from "../../types/infra";
import type { NodeLayout } from "../topology";

function worker(over: Partial<TopologyWorker> = {}): TopologyWorker {
  return {
    id: "w1",
    name: "cooja1",
    status: "available",
    enabled: true,
    held: false,
    held_kind: null,
    message: "ready",
    run: null,
    ...over,
  };
}

function dns(over: Partial<MasterDns> = {}): MasterDns {
  return {
    ok: false,
    message: "the DNS resolver did not answer",
    names: 2,
    failing: ["andromeda.example", "lab.example"],
    missing: [],
    self_restart: true,
    restart_waiting: null,
    ...over,
  };
}

function host(over: Partial<TopologyHost> = {}): TopologyHost {
  return {
    id: "h1",
    name: "local",
    type: "local",
    status: "online",
    enabled: true,
    schedulable: true,
    withdrawing: false,
    last_check: null,
    message: "",
    workers: [],
    ...over,
  };
}

describe("tones", () => {
  it("reads the master by whether its health cycle is writing", () => {
    expect(masterTone("alive")).toBe("ok");
    expect(masterTone("stale")).toBe("bad");
    expect(masterTone("unknown")).toBe("off");
  });

  it("reads a host by reachability, and disabled before anything else", () => {
    expect(hostTone(host())).toBe("ok");
    expect(hostTone(host({ status: "degraded" }))).toBe("warn");
    expect(hostTone(host({ status: "offline" }))).toBe("bad");
    expect(hostTone(host({ status: "misconfigured" }))).toBe("bad");
    expect(hostTone(host({ enabled: false, status: "online" }))).toBe("off");
  });

  it("flags a host being removed even while it is still online", () => {
    expect(hostTone(host({ withdrawing: true }))).toBe("warn");
  });

  it("shows a worker finishing a run as running, even if switched off", () => {
    // Saying "disabled" would hide the simulation still on it.
    expect(workerTone(worker({ status: "busy", enabled: false }))).toBe("busy");
    expect(workerLabel(worker({ status: "busy", enabled: false }))).toBe("running");
  });

  it("tells a held worker apart from one simply switched off", () => {
    expect(workerTone(worker({ status: "disabled", enabled: false, held: true }))).toBe("warn");
    expect(workerTone(worker({ status: "disabled", enabled: false }))).toBe("off");
  });

  it("reads a reservation as running — the container is already taken", () => {
    expect(workerTone(worker({ status: "reserved" }))).toBe("busy");
  });

  it("reads an offline or failing worker as bad", () => {
    expect(workerTone(worker({ status: "offline" }))).toBe("bad");
    expect(workerTone(worker({ status: "error" }))).toBe("bad");
  });
});

describe("layoutTopology", () => {
  const eight = Array.from({ length: 8 }, (_, i) => worker({ id: `a${i}` }));
  const topology = {
    hosts: [
      host({ id: "a", workers: eight }),
      host({ id: "b", workers: [worker({ id: "b0" }), worker({ id: "b1" })] }),
    ],
  };
  const find = (nodes: NodeLayout[], id: string): NodeLayout => nodes.find((n) => n.id === id)!;

  it("reads left to right: master, then hosts, then their workers", () => {
    const { nodes } = layoutTopology(topology);
    const master = find(nodes, MASTER_NODE_ID);
    const hostA = find(nodes, hostNodeId("a"));
    const worker0 = find(nodes, workerNodeId("a0"));

    expect(master.x).toBeLessThan(hostA.x);
    expect(hostA.x).toBeLessThan(worker0.x);
  });

  it("gives every host its own line, with its workers on it", () => {
    // A row of hosts across the top ran out of width at ten; a column of them
    // only grows downward.
    const { nodes } = layoutTopology(topology);
    const yA = find(nodes, hostNodeId("a")).y;
    const yB = find(nodes, hostNodeId("b")).y;

    expect(yA).not.toBe(yB);
    expect(new Set(nodes.filter((n) => n.hostId === "a").map((n) => n.y))).toEqual(new Set([yA]));
  });

  it("wraps a host with many workers onto a second line", () => {
    const many = Array.from({ length: WORKERS_PER_ROW + 3 }, (_, i) => worker({ id: `m${i}` }));
    const { nodes, rows } = layoutTopology({ hosts: [host({ id: "m", workers: many })] });

    expect(new Set(nodes.filter((n) => n.hostId === "m").map((n) => n.y)).size).toBe(2);
    expect(rows).toBe(2);
  });

  it("grows with the number of hosts, not their width", () => {
    const ten = Array.from({ length: 10 }, (_, i) =>
      host({ id: `h${i}`, workers: [worker({ id: `w${i}a` }), worker({ id: `w${i}b` })] }),
    );

    const { rows, bounds } = layoutTopology({ hosts: ten });

    expect(rows).toBe(10);
    expect(bounds.xMax - bounds.xMin).toBeLessThan(10);
  });

  it("centres the master on the hosts", () => {
    const { nodes, rows } = layoutTopology(topology);

    expect(find(nodes, MASTER_NODE_ID).y).toBe((rows - 1) / 2);
  });

  it("links the master to every host and every host to its workers", () => {
    const { edges } = layoutTopology(topology);

    expect(edges.filter((e) => e.source === MASTER_NODE_ID)).toHaveLength(2);
    expect(edges.filter((e) => e.source === hostNodeId("a"))).toHaveLength(8);
  });

  it("is the same picture every time — no layout drift between polls", () => {
    expect(layoutTopology(topology)).toEqual(layoutTopology(topology));
  });

  it("draws a host with no workers on a line of its own", () => {
    const { nodes, rows } = layoutTopology({ hosts: [host({ id: "x" })] });

    expect(nodes.map((n) => n.kind)).toEqual(["master", "host"]);
    expect(rows).toBe(1);
  });
});

describe("experimentColors", () => {
  it("keeps an experiment's colour when a newer one starts", () => {
    // Ids grow with creation time; the newer one takes the next colour.
    const before = experimentColors([{ id: "a1", name: "", on_workers: 1, by_host: {} }]);
    const after = experimentColors([
      { id: "b2", name: "", on_workers: 9, by_host: {} },
      { id: "a1", name: "", on_workers: 1, by_host: {} },
    ]);

    expect(after["a1"]).toBe(before["a1"]);
    expect(after["b2"]).not.toBe(after["a1"]);
  });

  it("never uses the status colours for an experiment", () => {
    expect(EXPERIMENT_COLORS).not.toContain("#3b82f6");
    expect(EXPERIMENT_COLORS).not.toContain("#dc2626");
  });
});

describe("tooltips", () => {
  const now = "2026-10-21T12:01:30Z";

  it("names the experiment, generation and how long the run has held the worker", () => {
    const text = workerTooltip(
      worker({
        status: "busy",
        run: {
          simulation_id: "s",
          experiment_id: "e",
          experiment_name: "pilot",
          generation_index: 3,
          random_seed: 4242,
          assigned_at: "2026-10-21T12:00:00Z",
        },
      }),
      now,
    );

    expect(text).toContain("experiment: pilot");
    expect(text).toContain("generation 3 · seed 4242");
    expect(text).toContain("1 min 30 s");
  });

  it("escapes names and messages before ECharts renders them as HTML", () => {
    const text = workerTooltip(
      worker({ name: "<img src=x onerror=alert(1)>", message: "a & b" }),
      now,
    );

    expect(text).not.toContain("<img");
    expect(text).toContain("&lt;img");
    expect(text).toContain("a &amp; b");
  });

  it("says plainly when the master-node has stopped checking", () => {
    const text = masterTooltip({
      status: "stale",
      last_seen: null,
      age_sec: 300,
      stale_after_sec: 120,
    });

    expect(text).toContain("not checking hosts");
    expect(text).toContain("5 min 0 s");
  });
});

describe("the master-node's DNS", () => {
  it("reads a running master whose resolver answers nothing as degraded", () => {
    expect(masterTone("alive", dns())).toBe("warn");
    expect(masterTone("alive", dns({ ok: true }))).toBe("ok");
    expect(masterTone("alive", null)).toBe("ok");
  });

  it("keeps a stopped master red whatever its last DNS report said", () => {
    expect(masterTone("stale", dns())).toBe("bad");
  });

  it("names the failing names and what is holding the restart back", () => {
    const text = masterTooltip({
      status: "alive",
      last_seen: null,
      age_sec: 4,
      stale_after_sec: 120,
      dns: dns({ restart_waiting: "will restart once no simulation is placed" }),
    });

    expect(text).toContain("DNS not answering");
    expect(text).toContain("andromeda.example, lab.example");
    expect(text).toContain("once no simulation is placed");
  });

  it("says to restart by hand when self-restart is off", () => {
    expect(dnsLines(dns({ self_restart: false })).join(" ")).toContain("restart the master-node");
  });

  it("says nothing about DNS when no host is reached by name", () => {
    expect(dnsLines(dns({ ok: true, names: 0, failing: [] }))).toEqual([]);
    expect(dnsLines(null)).toEqual([]);
  });

  it("escapes the names it repeats", () => {
    expect(dnsLines(dns({ failing: ["<b>x</b>"] })).join("")).toContain("&lt;b&gt;x");
  });
});

describe("hostTooltip", () => {
  const now = "2026-10-22T10:00:00Z";

  it("says when a host backing off will be checked again", () => {
    const text = hostTooltip(host({ retry_at: "2026-10-22T10:02:00Z" }), now);

    expect(text).toContain("backing off — next check in 2 min 0 s");
  });

  it("shows the last address a host reached by name answered on", () => {
    const text = hostTooltip(
      host({ address: "andromeda.example", last_known_ip: "143.107.0.10" }),
      now,
    );

    expect(text).toContain("andromeda.example (last reached at 143.107.0.10)");
  });

  it("does not repeat an address used directly", () => {
    const text = hostTooltip(host({ address: "10.0.0.1", last_known_ip: "10.0.0.1" }), now);

    expect(text).not.toContain("last reached");
  });
});

describe("time", () => {
  it("measures against the server's clock, never negative", () => {
    expect(elapsedSeconds("2026-10-21T12:00:00Z", "2026-10-21T12:00:42Z")).toBe(42);
    expect(elapsedSeconds("2026-10-21T12:00:10Z", "2026-10-21T12:00:00Z")).toBe(0);
    expect(elapsedSeconds(null, "2026-10-21T12:00:00Z")).toBeNull();
  });

  it("formats seconds, minutes and hours", () => {
    expect(formatElapsed(42)).toBe("42 s");
    expect(formatElapsed(125)).toBe("2 min 5 s");
    expect(formatElapsed(3720)).toBe("1 h 2 min");
    expect(formatElapsed(null)).toBe("—");
  });

  it("escapes every HTML-significant character", () => {
    expect(escapeHtml(`<a href="x">'&'</a>`)).toBe(
      "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;",
    );
  });
});

describe("summaries", () => {
  it("counts what is running on a host", () => {
    expect(
      hostLoad(
        host({ workers: [worker({ status: "busy" }), worker({ status: "reserved" }), worker()] }),
      ),
    ).toEqual({ running: 2, total: 3 });
  });

  it("counts the pool for the page header", () => {
    const result = summarize({
      hosts: [
        host({ workers: [worker({ status: "busy" }), worker()] }),
        host({ status: "offline", workers: [worker({ status: "offline" })] }),
      ],
    });

    expect(result).toEqual({ hosts: 2, hostsReachable: 1, workers: 3, running: 1, free: 1 });
  });
});
