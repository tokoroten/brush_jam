/**
 * The history strip: every AI result this room has made, newest first.
 *
 * The server keeps them (see apps/brushjam/src/brushjam/history.py); this is
 * only the state of looking at them. It is a plain reducer rather than a pile
 * of useState calls because the awkward parts - a listing that arrives after
 * the entry it was going to select was evicted, a refresh that lands out of
 * order - are exactly the parts worth testing.
 */

import type { HistoryEntry, HistoryListing } from '@brushjam/shared';
import { clampPresetDenoise, type PresetTarget } from './presetPicker.js';
import type { SharedDraft } from './sharedDraft.js';

export interface GalleryState {
  /** The strip is shown. Nothing is fetched while it is closed. */
  open: boolean;
  /** Newest first, as the server returns them. */
  entries: HistoryEntry[];
  /** The entry shown large, by number; null means the strip only. */
  selected: number | null;
  loading: boolean;
  /** True once the server has answered at least once. */
  loaded: boolean;
  /** The server was started with HISTORY_ENABLED=0. */
  enabled: boolean;
  error: string | null;
  /**
   * The newest entry number the server has announced through `ai_result`,
   * whether or not it has been fetched yet. This is what says a refresh is
   * due, so a result that arrives while the strip is closed is not missed.
   */
  latestN: number | null;
}

export const initialGallery: GalleryState = {
  open: false,
  entries: [],
  selected: null,
  loading: false,
  loaded: false,
  enabled: true,
  error: null,
  latestN: null,
};

export const toggleGallery = (state: GalleryState): GalleryState => ({
  ...state,
  open: !state.open,
  // Closing puts the large view away too: reopening should show the strip.
  selected: state.open ? null : state.selected,
});

export const galleryLoading = (state: GalleryState): GalleryState => ({
  ...state,
  loading: true,
  error: null,
});

/**
 * A listing arrived.
 *
 * The selection survives only if the entry is still there: the store evicts
 * oldest first, and a selection pointing at an evicted entry would show a
 * broken image with somebody else's settings under it.
 */
export function galleryLoaded(state: GalleryState, listing: HistoryListing): GalleryState {
  const entries = [...(listing.entries ?? [])].sort((a, b) => b.n - a.n);
  const newest = entries.length > 0 ? entries[0]!.n : null;
  return {
    ...state,
    entries,
    enabled: listing.enabled !== false,
    loading: false,
    loaded: true,
    error: null,
    selected: entries.some((e) => e.n === state.selected) ? state.selected : null,
    // Never moves backwards: a slow listing must not un-announce a newer
    // result that arrived while it was in flight.
    latestN: newest === null ? state.latestN : Math.max(newest, state.latestN ?? newest),
  };
}

export const galleryFailed = (state: GalleryState, message: string): GalleryState => ({
  ...state,
  loading: false,
  error: message,
});

/** The server saved a new result under this number (`ai_result.historyN`). */
export const galleryAnnounced = (state: GalleryState, n: number | null | undefined): GalleryState =>
  typeof n !== 'number' || (state.latestN !== null && n <= state.latestN)
    ? state
    : { ...state, latestN: n };

/** Clicking a thumbnail; clicking the open one again puts it away. */
export const selectEntry = (state: GalleryState, n: number | null): GalleryState => ({
  ...state,
  selected: state.selected === n ? null : n,
});

export const selectedEntry = (state: GalleryState): HistoryEntry | null =>
  state.entries.find((e) => e.n === state.selected) ?? null;

/**
 * The single rule the effect in Room.tsx follows: fetch when the strip is
 * open and either has never been filled or is older than what the room has
 * announced. A closed gallery fetches nothing, so a room can run all afternoon
 * with nobody looking at the history and never ask the server for it.
 */
export function shouldFetch(state: GalleryState): boolean {
  if (!state.open || state.loading) return false;
  if (!state.loaded) return true;
  const newest = state.entries.length > 0 ? state.entries[0]!.n : null;
  return state.latestN !== null && (newest === null || state.latestN > newest);
}

/** The fields "use these settings" writes back into the room. */
export interface HistoryTarget extends PresetTarget {
  seed: SharedDraft<number>;
}

/**
 * "use these settings": put an old result's prompt, negative prompt, denoise
 * and seed back into the room.
 *
 * The same one-shot fill as a preset, and sent immediately for the same
 * reason. The profile and the resolution are deliberately not applied: they
 * are what the room's hardware can do right now, not part of the look, and a
 * `quality` entry replayed on a worker that only has `fast` would be a setting
 * the server refuses.
 */
export function applyHistorySettings(entry: HistoryEntry, target: HistoryTarget): void {
  target.prompt.set(entry.prompt ?? '');
  target.prompt.flush();
  target.negative.set(entry.negativePrompt ?? '');
  target.negative.flush();
  if (typeof entry.denoise === 'number') {
    target.denoise.set(clampPresetDenoise(entry.denoise, target.maxDenoise));
    target.denoise.flush();
  }
  if (typeof entry.seed === 'number' && Number.isFinite(entry.seed)) {
    target.seed.set(Math.floor(entry.seed));
    target.seed.flush();
  }
}

/** "2.1 s", the way the status pill says it. */
export const formatLatency = (ms: number): string =>
  !Number.isFinite(ms) || ms < 0 ? '' : ms >= 10_000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`;

export function formatTime(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  const date = new Date(ms);
  const two = (n: number): string => String(n).padStart(2, '0');
  return `${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`;
}
