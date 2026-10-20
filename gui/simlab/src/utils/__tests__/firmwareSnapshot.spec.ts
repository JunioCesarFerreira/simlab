import { describe, expect, it } from "vitest";
import {
  fileCount,
  fileExtension,
  formatBytes,
  hasFiles,
  missingFileCount,
  optionsLabel,
  shortHash,
  statusInfo,
  totalBytes,
} from "../firmwareSnapshot";
import type { FirmwareSnapshotDto } from "../../types/simlab";

function snapshot(
  overrides: Partial<FirmwareSnapshotDto> = {},
): FirmwareSnapshotDto {
  return {
    status: "captured",
    captured_at: "2024-01-01T10:30:00",
    schema_version: 1,
    repositories: [
      {
        option_keys: ["csma"],
        source_repository_id: "repo-a",
        name: "rpl-udp-csma",
        description: "",
        files: [
          { file_name: "node.c", file_id: "f1", size_bytes: 1000, sha256: "a".repeat(64) },
          { file_name: "Makefile", file_id: "f2", size_bytes: 24 },
        ],
        missing_files: [],
      },
      {
        option_keys: ["tsch", "tsch-6o"],
        source_repository_id: "repo-b",
        name: "rpl-udp-tsch",
        description: "",
        files: [{ file_name: "root.c", file_id: "f3", size_bytes: 500 }],
        missing_files: [{ file_name: "gone.c", origin_file_id: "f9" }],
      },
    ],
    ...overrides,
  };
}

describe("statusInfo", () => {
  it("reports a complete capture as ok", () => {
    const info = statusInfo(snapshot());
    expect(info.label).toBe("Captured");
    expect(info.tone).toBe("ok");
  });

  it("warns when the capture is partial", () => {
    expect(statusInfo(snapshot({ status: "partial" })).tone).toBe("warn");
  });

  it("treats a failed capture as an error", () => {
    expect(statusInfo(snapshot({ status: "failed" })).tone).toBe("error");
  });

  it("prefers the backend reason over the generic message", () => {
    const info = statusInfo(
      snapshot({ status: "skipped", reason: "Synthetic run, no firmware." }),
    );
    expect(info.message).toBe("Synthetic run, no firmware.");
    expect(info.tone).toBe("muted");
  });

  it("prefers the backend error over the generic message", () => {
    const info = statusInfo(snapshot({ status: "failed", error: "GridFS down" }));
    expect(info.message).toBe("GridFS down");
  });

  it("falls back for a missing or unrecognized status", () => {
    expect(statusInfo(null).label).toBe("Unknown");
    expect(
      statusInfo({ status: "weird" as never, repositories: [] }).label,
    ).toBe("Unknown");
  });
});

describe("counting", () => {
  it("counts captured files across repositories", () => {
    expect(fileCount(snapshot())).toBe(3);
  });

  it("counts files that could not be copied separately", () => {
    // Missing files must never inflate the "preserved" count.
    expect(missingFileCount(snapshot())).toBe(1);
  });

  it("sums sizes, treating an unrecorded size as zero", () => {
    expect(totalBytes(snapshot())).toBe(1524);
  });

  it("returns zero for an absent or empty snapshot", () => {
    expect(fileCount(null)).toBe(0);
    expect(missingFileCount(undefined)).toBe(0);
    expect(totalBytes(snapshot({ repositories: [] }))).toBe(0);
  });

  it("reports whether anything can be viewed or downloaded", () => {
    expect(hasFiles(snapshot())).toBe(true);
    expect(hasFiles(snapshot({ status: "skipped", repositories: [] }))).toBe(false);
    expect(hasFiles(null)).toBe(false);
  });
});

describe("formatBytes", () => {
  it.each([
    [0, "0 B"],
    [512, "512 B"],
    [2048, "2.0 KB"],
    [5 * 1024 * 1024, "5.0 MB"],
  ])("formats %i bytes as %s", (input, expected) => {
    expect(formatBytes(input)).toBe(expected);
  });

  it("renders a dash for an unknown size", () => {
    expect(formatBytes(undefined)).toBe("—");
  });
});

describe("shortHash", () => {
  it("keeps the first 12 characters", () => {
    expect(shortHash("abcdef0123456789")).toBe("abcdef012345");
  });

  it("renders a dash when there is no digest", () => {
    expect(shortHash(undefined)).toBe("—");
  });
});

describe("optionsLabel", () => {
  it("joins every option a repository was selected for", () => {
    expect(optionsLabel(snapshot().repositories[1]!)).toBe("tsch, tsch-6o");
  });

  it("is empty when no option key was recorded", () => {
    expect(
      optionsLabel({ ...snapshot().repositories[0]!, option_keys: [] }),
    ).toBe("");
  });
});

describe("fileExtension", () => {
  it.each([
    ["node.c", "c"],
    ["project-conf.h", "h"],
    ["metrics.PACKET.H", "h"],
  ])("derives %s as .%s", (name, expected) => {
    expect(fileExtension(name)).toBe(expected);
  });

  it("falls back to txt for extension-less firmware files", () => {
    // Makefile has no suffix, but /files/{id}/as/{ext} always needs one.
    expect(fileExtension("Makefile")).toBe("txt");
    expect(fileExtension(".gitignore")).toBe("txt");
    expect(fileExtension("node.")).toBe("txt");
  });
});
