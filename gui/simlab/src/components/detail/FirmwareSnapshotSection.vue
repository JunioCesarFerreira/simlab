<template>
  <div class="card fw-card">
    <div class="section-title fw-title">
      <span>
        Firmware
        <span :class="['fw-status', `fw-status--${info.tone}`]">{{ info.label }}</span>
      </span>
      <button
        v-if="downloadable"
        class="fw-btn"
        :disabled="downloading"
        @click="downloadZip"
      >
        {{ downloading ? "Preparing…" : "Download ZIP" }}
      </button>
    </div>

    <p :class="['fw-note', `fw-note--${info.tone}`]">{{ info.message }}</p>
    <p v-if="downloadError" class="fw-note fw-note--error">{{ downloadError }}</p>

    <template v-if="downloadable">
      <div class="fw-summary">
        <span><strong>{{ totalFiles }}</strong> file{{ totalFiles === 1 ? "" : "s" }}</span>
        <span class="fw-sep">·</span>
        <span
          ><strong>{{ snapshot!.repositories.length }}</strong> repositor{{
            snapshot!.repositories.length === 1 ? "y" : "ies"
          }}</span
        >
        <span class="fw-sep">·</span>
        <span>{{ formatBytes(totalSize) }}</span>
        <template v-if="snapshot!.captured_at">
          <span class="fw-sep">·</span>
          <span>captured {{ formatDate(snapshot!.captured_at) }}</span>
        </template>
      </div>

      <details
        v-for="(repo, index) in snapshot!.repositories"
        :key="repo.source_repository_id || index"
        class="fw-repo"
        :open="snapshot!.repositories.length === 1"
      >
        <summary class="fw-repo-head">
          <span class="fw-repo-name">{{ repo.name || "(unnamed repository)" }}</span>
          <span v-if="optionsLabel(repo)" class="fw-chip">{{ optionsLabel(repo) }}</span>
          <span class="fw-count">{{ repo.files.length }}</span>
        </summary>

        <p v-if="repo.description" class="fw-repo-desc">{{ repo.description }}</p>
        <RouterLink
          v-if="repo.source_repository_id"
          class="fw-origin"
          :to="`/sources/${repo.source_repository_id}`"
        >
          Original repository ↗
        </RouterLink>

        <div v-if="repo.files.length" class="fw-table">
          <div class="fw-table-head">
            <span>File</span>
            <span>Size</span>
            <span>sha256</span>
            <span></span>
          </div>
          <div v-for="file in repo.files" :key="file.file_id" class="fw-row">
            <button
              class="fw-name-btn"
              :title="`View ${file.file_name}`"
              @click="openViewer(file)"
            >
              <span class="fw-icon">📄</span>{{ file.file_name }}
            </button>
            <span class="fw-dim">{{ formatBytes(file.size_bytes) }}</span>
            <span class="fw-dim mono" :title="file.sha256">{{ shortHash(file.sha256) }}</span>
            <span class="fw-row-actions">
              <button class="fw-icon-btn" title="View file" @click="openViewer(file)">
                &#128065;
              </button>
              <button
                class="fw-icon-btn"
                title="Download file"
                :disabled="downloadingFileId === file.file_id"
                @click="downloadOne(file)"
              >
                &#8623;
              </button>
            </span>
          </div>
        </div>

        <p v-if="repo.missing_files.length" class="fw-note fw-note--warn">
          {{ repo.missing_files.length }} file(s) could not be copied:
          {{ repo.missing_files.map((f) => f.file_name || "(unnamed)").join(", ") }}
        </p>
      </details>
    </template>

    <!-- Viewer reuses the source-repository highlighter with a snapshot loader -->
    <SourceFileViewer
      v-if="viewerFile"
      :file-id="viewerFile.id"
      :file-name="viewerFile.name"
      :load="loadFileContent"
      @close="viewerFile = null"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { RouterLink } from "vue-router";
import SourceFileViewer from "../sources/SourceFileViewer.vue";
import { getFirmwareFileContent } from "../../api/experiments";
import { downloadFile, downloadFirmwareZip } from "../../api/files";
import type { FirmwareFileDto, FirmwareSnapshotDto } from "../../types/simlab";
import {
  fileCount,
  fileExtension,
  formatBytes,
  hasFiles,
  optionsLabel,
  shortHash,
  statusInfo,
  totalBytes,
} from "../../utils/firmwareSnapshot";

const props = defineProps<{
  experimentId: string;
  /** Null/undefined for runs that predate firmware tracking, or not started. */
  snapshot?: FirmwareSnapshotDto | null;
  /** Whether the experiment has already started, which is when firmware is copied. */
  started?: boolean;
}>();

const info = computed(() => {
  if (props.snapshot) return statusInfo(props.snapshot);
  return props.started
    ? {
        label: "Not recorded",
        message:
          "This run has no firmware record — it started before firmware tracking existed.",
        tone: "muted" as const,
      }
    : {
        label: "Pending",
        message: "The firmware is copied when the experiment starts.",
        tone: "muted" as const,
      };
});

const totalFiles = computed(() => fileCount(props.snapshot));
const totalSize = computed(() => totalBytes(props.snapshot));
const downloadable = computed(() => hasFiles(props.snapshot));

// ── downloads ───────────────────────────────────────────────────────────────

const downloading = ref(false);
const downloadingFileId = ref<string | null>(null);
const downloadError = ref("");

async function downloadZip() {
  downloading.value = true;
  downloadError.value = "";
  try {
    await downloadFirmwareZip(props.experimentId);
  } catch (e) {
    downloadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    downloading.value = false;
  }
}

async function downloadOne(file: FirmwareFileDto) {
  if (!file.file_id) return;
  downloadingFileId.value = file.file_id;
  downloadError.value = "";
  try {
    await downloadFile(file.file_id, fileExtension(file.file_name));
  } catch (e) {
    downloadError.value = e instanceof Error ? e.message : String(e);
  } finally {
    downloadingFileId.value = null;
  }
}

// ── viewer ──────────────────────────────────────────────────────────────────

const viewerFile = ref<{ id: string; name: string } | null>(null);

function openViewer(file: FirmwareFileDto) {
  if (!file.file_id) return;
  viewerFile.value = { id: file.file_id, name: file.file_name };
}

function loadFileContent(fileId: string): Promise<string> {
  return getFirmwareFileContent(props.experimentId, fileId);
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
</script>

<style scoped>
.fw-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.fw-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.fw-status {
  margin-left: 8px;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.fw-status--ok {
  color: var(--status-done, #15803d);
  background: rgba(21, 128, 61, 0.1);
}

.fw-status--warn {
  color: var(--status-warning, #b45309);
  background: rgba(217, 119, 6, 0.12);
}

.fw-status--error {
  color: var(--status-error);
  background: rgba(220, 38, 38, 0.1);
}

.fw-status--muted {
  color: var(--color-text-muted);
  background: var(--color-bg);
}

.fw-note {
  margin: 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.fw-note--warn {
  color: var(--status-warning, #b45309);
}

.fw-note--error {
  color: var(--status-error);
}

.fw-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.fw-sep {
  opacity: 0.5;
}

.fw-btn {
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text);
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  cursor: pointer;
}

.fw-btn:disabled {
  opacity: 0.6;
  cursor: default;
}

.fw-repo {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 8px 12px;
  background: var(--color-bg);
}

.fw-repo-head {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
}

.fw-repo-name {
  flex: 1;
}

.fw-chip {
  font-size: 11px;
  font-weight: 500;
  padding: 1px 8px;
  border-radius: 999px;
  color: var(--color-text-muted);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
}

.fw-count {
  font-size: 11px;
  color: var(--color-text-muted);
}

.fw-repo-desc {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.fw-origin {
  display: inline-block;
  margin-top: 6px;
  font-size: 12px;
  color: var(--color-accent, #2563eb);
  text-decoration: none;
}

.fw-table {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
}

.fw-table-head,
.fw-row {
  display: grid;
  grid-template-columns: 1fr 80px 120px 72px;
  gap: 8px;
  align-items: center;
  padding: 6px 0;
}

.fw-table-head {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
  border-bottom: 1px solid var(--color-border);
}

.fw-row + .fw-row {
  border-top: 1px solid var(--color-border);
}

.fw-name-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0;
  font-size: 13px;
  text-align: left;
  color: var(--color-text);
  background: none;
  border: none;
  cursor: pointer;
  overflow: hidden;
  text-overflow: ellipsis;
}

.fw-name-btn:hover {
  text-decoration: underline;
}

.fw-dim {
  font-size: 12px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
}

.mono {
  font-family: var(--font-mono, ui-monospace, monospace);
}

.fw-row-actions {
  display: flex;
  gap: 4px;
  justify-content: flex-end;
}

.fw-icon-btn {
  padding: 2px 6px;
  font-size: 12px;
  color: var(--color-text-muted);
  background: none;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm, 4px);
  cursor: pointer;
}

.fw-icon-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
