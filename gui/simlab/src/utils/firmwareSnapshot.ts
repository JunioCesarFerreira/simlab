/**
 * Presentation helpers for the firmware snapshot captured at experiment start.
 *
 * Kept out of the component so the rules that decide what an operator reads —
 * "is this run's firmware fully preserved?" — are testable on their own.
 */
import type {
  FirmwareRepositorySnapshotDto,
  FirmwareSnapshotDto,
  FirmwareSnapshotStatus,
} from "../types/simlab";

export interface FirmwareStatusInfo {
  /** Short badge text. */
  label: string;
  /** Sentence explaining what the status means for traceability. */
  message: string;
  /** Severity driving the badge/alert colour. */
  tone: "ok" | "warn" | "error" | "muted";
}

const STATUS_INFO: Record<FirmwareSnapshotStatus, FirmwareStatusInfo> = {
  captured: {
    label: "Captured",
    message:
      "Every firmware file used by this experiment was copied and is preserved here.",
    tone: "ok",
  },
  partial: {
    label: "Partial",
    message:
      "Some firmware files could not be copied — the record below is incomplete.",
    tone: "warn",
  },
  capturing: {
    label: "Capturing…",
    message: "The firmware copy is still in progress.",
    tone: "muted",
  },
  skipped: {
    label: "Not applicable",
    message: "This experiment references no firmware.",
    tone: "muted",
  },
  failed: {
    label: "Failed",
    message:
      "The firmware could not be captured, so this run has no firmware record.",
    tone: "error",
  },
};

const UNKNOWN_STATUS: FirmwareStatusInfo = {
  label: "Unknown",
  message: "Unrecognized firmware snapshot status.",
  tone: "muted",
};

export function statusInfo(
  snapshot: FirmwareSnapshotDto | null | undefined,
): FirmwareStatusInfo {
  if (!snapshot) return UNKNOWN_STATUS;
  const info = STATUS_INFO[snapshot.status];
  if (!info) return UNKNOWN_STATUS;
  // A backend-provided reason/error is more specific than the generic text.
  const detail = snapshot.reason || snapshot.error;
  return detail ? { ...info, message: detail } : info;
}

/** Number of files actually preserved (missing ones are not counted). */
export function fileCount(
  snapshot: FirmwareSnapshotDto | null | undefined,
): number {
  return (snapshot?.repositories ?? []).reduce(
    (total, repo) => total + (repo.files?.length ?? 0),
    0,
  );
}

export function missingFileCount(
  snapshot: FirmwareSnapshotDto | null | undefined,
): number {
  return (snapshot?.repositories ?? []).reduce(
    (total, repo) => total + (repo.missing_files?.length ?? 0),
    0,
  );
}

/** Total captured bytes; files without a recorded size contribute zero. */
export function totalBytes(
  snapshot: FirmwareSnapshotDto | null | undefined,
): number {
  return (snapshot?.repositories ?? []).reduce(
    (total, repo) =>
      total +
      (repo.files ?? []).reduce((sum, f) => sum + (f.size_bytes ?? 0), 0),
    0,
  );
}

/** True when there is at least one file to view or download. */
export function hasFiles(
  snapshot: FirmwareSnapshotDto | null | undefined,
): boolean {
  return fileCount(snapshot) > 0;
}

export function formatBytes(bytes: number | undefined): string {
  if (bytes === undefined || bytes === null || Number.isNaN(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** First 12 hex characters — enough to compare two builds by eye. */
export function shortHash(sha256: string | undefined): string {
  return sha256 ? sha256.slice(0, 12) : "—";
}

/** Label for the options (MAC protocols) this repository was selected for. */
export function optionsLabel(repo: FirmwareRepositorySnapshotDto): string {
  return (repo.option_keys ?? []).join(", ");
}

/** File extension used when downloading a copy through /files/{id}/as/{ext}. */
export function fileExtension(fileName: string): string {
  const base = fileName.split("/").pop() ?? fileName;
  const dot = base.lastIndexOf(".");
  // Extension-less firmware files (Makefile) still need a suffix for the
  // download route; plain text keeps them readable in the browser.
  if (dot <= 0 || dot === base.length - 1) return "txt";
  return base.slice(dot + 1).toLowerCase();
}
