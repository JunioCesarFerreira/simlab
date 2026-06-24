import type { ProblemDraft } from '../types/problem'

const STORAGE_KEY = 'simlab-problem-draft'

export function saveDraft(draft: ProblemDraft): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(draft))
  } catch {
    // Ignore storage errors (e.g. private browsing quota)
  }
}

export function loadDraft(): ProblemDraft | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as ProblemDraft
  } catch {
    return null
  }
}

export function clearDraft(): void {
  localStorage.removeItem(STORAGE_KEY)
}
