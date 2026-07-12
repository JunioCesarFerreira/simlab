import type { ProblemDraft } from '../types/problem'

const STORAGE_KEY = 'simlab-problem-draft'

/**
 * Bump whenever ProblemDraft changes shape in a way a stored draft can't
 * survive (renamed/retyped fields, new required nested structure), and add a
 * migration step below. Adding an optional top-level field does NOT require a
 * bump — the store shallow-merges defaults for those.
 */
export const DRAFT_SCHEMA_VERSION = 1

interface StoredDraft {
  version: number
  draft: ProblemDraft
}

/**
 * Sequential migrations: MIGRATIONS[n] upgrades a draft from version n to
 * n + 1. A draft that can't be migrated to the current version is discarded
 * (returning null) rather than loaded half-broken.
 */
const MIGRATIONS: Record<number, (draft: ProblemDraft) => ProblemDraft> = {
  // Example for a future bump to version 2:
  // 1: (draft) => ({ ...draft, newRequiredField: defaultValue }),
}

function migrate(stored: StoredDraft): ProblemDraft | null {
  let { version, draft } = stored
  while (version < DRAFT_SCHEMA_VERSION) {
    const step = MIGRATIONS[version]
    if (!step) return null
    draft = step(draft)
    version++
  }
  return draft
}

export function saveDraft(draft: ProblemDraft): void {
  try {
    const stored: StoredDraft = { version: DRAFT_SCHEMA_VERSION, draft }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored))
  } catch {
    // Ignore storage errors (e.g. private browsing quota)
  }
}

export function loadDraft(): ProblemDraft | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as StoredDraft | ProblemDraft
    // Drafts saved before versioning were the bare ProblemDraft — treat them
    // as version 1, which is the shape they had.
    const stored: StoredDraft =
      typeof parsed === 'object' && parsed !== null && 'version' in parsed && 'draft' in parsed
        ? (parsed as StoredDraft)
        : { version: 1, draft: parsed as ProblemDraft }
    return migrate(stored)
  } catch {
    return null
  }
}

export function clearDraft(): void {
  localStorage.removeItem(STORAGE_KEY)
}
