// @vitest-environment vue-renderer
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRenderer, defineComponent, h, nextTick } from "vue";
import FirmwareSnapshotSection from "../FirmwareSnapshotSection.vue";
import type { FirmwareSnapshotDto } from "../../../types/simlab";

const api = vi.hoisted(() => ({
  getFirmwareFileContent: vi.fn(async () => "int main(void) {}"),
  downloadFile: vi.fn(async () => {}),
  downloadFirmwareZip: vi.fn(async () => {}),
}));
vi.mock("../../../api/experiments", () => ({
  getFirmwareFileContent: api.getFirmwareFileContent,
}));
vi.mock("../../../api/files", () => ({
  downloadFile: api.downloadFile,
  downloadFirmwareZip: api.downloadFirmwareZip,
}));
// RouterLink needs a router instance we do not have here; the viewer teleports
// to a DOM body the custom renderer does not provide. Both are stubbed so the
// section itself is what gets exercised.
vi.mock("vue-router", () => ({
  RouterLink: (props: { to?: string }, ctx: { slots: { default?: () => unknown } }) =>
    h("a", { href: props.to }, ctx.slots.default?.() as never),
}));
const viewerProps = vi.hoisted(() => ({ current: null as Record<string, unknown> | null }));
vi.mock("../../sources/SourceFileViewer.vue", () => {
  const stub = (props: Record<string, unknown>) => {
    viewerProps.current = props;
    return h("div", { class: "viewer-stub" });
  };
  // Declared so the stub receives the real prop names, not raw attrs.
  stub.props = ["fileId", "fileName", "load"];
  return { default: stub };
});

// Mount the real component into an in-memory host so clicks and rendered text
// can be asserted without a browser.
interface HostNode {
  tag: string;
  parent: HostNode | null;
  children: HostNode[];
  text: string;
  props: Record<string, unknown>;
}
const node = (tag = "", text = ""): HostNode => ({
  tag, parent: null, children: [], text, props: {},
});
function remove(child: HostNode) {
  const siblings = child.parent?.children;
  if (siblings) siblings.splice(siblings.indexOf(child), 1);
  child.parent = null;
}
const renderer = createRenderer<HostNode, HostNode>({
  createElement: (tag) => node(String(tag)),
  createText: (text) => node("#text", text),
  createComment: () => node("#comment"),
  setText: (el, text) => { el.text = text; },
  setElementText: (el, text) => { el.text = text; },
  parentNode: (el) => el.parent,
  nextSibling: (el) =>
    el.parent?.children[el.parent.children.indexOf(el) + 1] ?? null,
  patchProp: (el, key, _previous, value) => { el.props[key] = value; },
  remove,
  insert: (el, parent, anchor) => {
    if (el.parent) remove(el);
    el.parent = parent;
    parent.children.splice(
      anchor ? parent.children.indexOf(anchor) : parent.children.length, 0, el,
    );
  },
});

function textOf(el: HostNode): string {
  return (el.text + el.children.map(textOf).join(" ")).replace(/\s+/g, " ").trim();
}
function walk(el: HostNode): HostNode[] {
  return [el, ...el.children.flatMap(walk)];
}
function buttonWithText(root: HostNode, label: string): HostNode {
  const match = walk(root).find(
    (el) => el.tag === "button" && textOf(el).includes(label),
  );
  if (!match) throw new Error(`no button matching ${label}`);
  return match;
}
function click(el: HostNode): void {
  (el.props.onClick as () => void)();
}

const snapshot: FirmwareSnapshotDto = {
  status: "captured",
  captured_at: "2024-01-01T10:30:00",
  schema_version: 1,
  repositories: [{
    option_keys: ["csma"],
    source_repository_id: "507f1f77bcf86cd799439021",
    name: "rpl-udp-csma",
    description: "CSMA firmware",
    files: [
      { file_name: "node.c", file_id: "f1", size_bytes: 2048, sha256: "abcdef0123456789" },
      { file_name: "Makefile", file_id: "f2", size_bytes: 24 },
    ],
    missing_files: [],
  }],
};

const unmounts: (() => void)[] = [];

interface SectionProps {
  experimentId: string;
  snapshot?: FirmwareSnapshotDto | null;
  started?: boolean;
}

function mount(props: SectionProps): HostNode {
  const root = node("root");
  const app = renderer.createApp(
    defineComponent({ setup: () => () => h(FirmwareSnapshotSection, props) }),
  );
  app.mount(root);
  unmounts.push(() => app.unmount());
  return root;
}

beforeEach(() => {
  vi.clearAllMocks();
  viewerProps.current = null;
});
afterEach(() => unmounts.splice(0).forEach((unmount) => unmount()));

describe("FirmwareSnapshotSection", () => {
  it("lists every captured file with its size and digest", () => {
    const root = mount({ experimentId: "exp-1", snapshot, started: true });
    const rendered = textOf(root);

    expect(rendered).toContain("Captured");
    expect(rendered).toContain("rpl-udp-csma");
    expect(rendered).toContain("node.c");
    expect(rendered).toContain("Makefile");
    expect(rendered).toContain("2.0 KB");
    expect(rendered).toContain("abcdef012345");
    expect(rendered).toContain("2 files");
    expect(rendered).toContain("1 repository");
  });

  it("opens the viewer scoped to this experiment's snapshot", async () => {
    const root = mount({ experimentId: "exp-1", snapshot, started: true });

    click(buttonWithText(root, "node.c"));
    await nextTick();

    expect(viewerProps.current).toMatchObject({ fileId: "f1", fileName: "node.c" });
    // The injected loader must hit the experiment-scoped endpoint, not the
    // shared source repository one.
    await (viewerProps.current!.load as (id: string) => Promise<string>)("f1");
    expect(api.getFirmwareFileContent).toHaveBeenCalledWith("exp-1", "f1");
  });

  it("downloads the whole snapshot as a ZIP", async () => {
    const root = mount({ experimentId: "exp-1", snapshot, started: true });

    click(buttonWithText(root, "Download ZIP"));
    await nextTick();

    expect(api.downloadFirmwareZip).toHaveBeenCalledWith("exp-1");
  });

  it("surfaces a failed download instead of silently doing nothing", async () => {
    api.downloadFirmwareZip.mockRejectedValueOnce(new Error("network down"));
    const root = mount({ experimentId: "exp-1", snapshot, started: true });

    click(buttonWithText(root, "Download ZIP"));
    await nextTick();
    await nextTick();

    expect(textOf(root)).toContain("network down");
  });

  it("downloads a single file with an extension the API accepts", async () => {
    const root = mount({ experimentId: "exp-1", snapshot, started: true });
    const rows = walk(root).filter(
      (el) => el.tag === "button" && el.props.title === "Download file",
    );

    click(rows[0]!);
    click(rows[1]!);
    await nextTick();

    expect(api.downloadFile).toHaveBeenNthCalledWith(1, "f1", "c");
    // Makefile has no extension; the route still needs one.
    expect(api.downloadFile).toHaveBeenNthCalledWith(2, "f2", "txt");
  });

  it("reports files that could not be copied", () => {
    const partial: FirmwareSnapshotDto = {
      ...snapshot,
      status: "partial",
      repositories: [{
        ...snapshot.repositories[0]!,
        missing_files: [{ file_name: "root.c", origin_file_id: "f9" }],
      }],
    };
    const rendered = textOf(mount({ experimentId: "exp-1", snapshot: partial, started: true }));

    expect(rendered).toContain("Partial");
    expect(rendered).toContain("root.c");
  });

  it("explains a skipped capture using the backend reason", () => {
    const skipped: FirmwareSnapshotDto = {
      status: "skipped",
      repositories: [],
      reason: "Experiment references no source repository.",
    };
    const root = mount({ experimentId: "exp-1", snapshot: skipped, started: true });

    expect(textOf(root)).toContain("Experiment references no source repository.");
    expect(() => buttonWithText(root, "Download ZIP")).toThrow();
  });

  it("distinguishes a run that predates tracking from one not started yet", () => {
    const started = textOf(mount({ experimentId: "exp-1", snapshot: null, started: true }));
    expect(started).toContain("Not recorded");

    const pending = textOf(mount({ experimentId: "exp-2", snapshot: null, started: false }));
    expect(pending).toContain("Pending");
    expect(pending).toContain("copied when the experiment starts");
  });
});
