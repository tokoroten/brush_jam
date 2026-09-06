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
  /** The announcement the request in flight set out to satisfy. */
  requestedN: number | null;
  /**
   * The announcement the last successful listing answered - which is not the
   * same as the newest entry it contained. A result the store has already
   * evicted is announced and then never appears in any listing, and comparing
   * against the entries meant `latestN` stayed permanently ahead and the strip
   * refetched forever.
   */
  satisfiedN: number | null;
  /** Consecutive failures, which is what sets the backoff. */
  failures: number;
  /** Nothing is fetched before this time (`Date.now()`); null means now. */
  retryAt: number | null;
}

/** First retry a second after a failure, doubling to half a minute. */
export const GALLERY_RETRY_MIN_MS = 1_000;
export const GALLERY_RETRY_MAX_MS = 30_000;

export const galleryBackoffMs = (failures: number): number =>
  Math.min(GALLERY_RETRY_MAX_MS, GALLERY_RETRY_MIN_MS * 2 ** Math.max(0, failures - 1));

export const initialGallery: GalleryState = {
  open: false,
  entries: [],
  selected: null,
  loading: false,
  loaded: false,
  enabled: true,
  error: null,
  latestN: null,
  requestedN: null,
  satisfiedN: null,
  failures: 0,
  retryAt: null,
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
  // Whatever this request comes back with, it will have answered the
  // announcement standing when it left.
  requestedN: state.latestN,
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
  const answered = [state.satisfiedN, state.requestedN, newest].filter(
    (n): n is number => n !== null,
  );
  return {
    ...state,
    entries,
    enabled: listing.enabled !== false,
    loading: false,
    loaded: true,
    error: null,
    failures: 0,
    retryAt: null,
    // What this listing proves has been answered: the announcement it set out
    // to satisfy, and anything newer it actually returned. An announcement
    // that arrived while it was in flight is not covered unless it is here.
    satisfiedN: answered.length === 0 ? state.satisfiedN : Math.max(...answered),
    selected: entries.some((e) => e.n === state.selected) ? state.selected : null,
    // Never moves backwards: a slow listing must not un-announce a newer
    // result that arrived while it was in flight.
    latestN: newest === null ? state.latestN : Math.max(newest, state.latestN ?? newest),
  };
}

/**
 * The request failed.
 *
 * `loading` goes back down, which used to be the whole story - and left
 * `shouldFetch` true, so the effect started another request in the same tick.
 * A server that is down therefore got a continuous stream of requests from
 * every open strip. The next attempt now waits out a doubling backoff.
 */
export const galleryFailed = (
  state: GalleryState,
  message: string,
  now: number = Date.now(),
): GalleryState => ({
  ...state,
  loading: false,
  error: message,
  failures: state.failures + 1,
  retryAt: now + galleryBackoffMs(state.failures + 1),
});

/** The backoff has run out: the next fetch may go. */
export const galleryRetryDue = (state: GalleryState): GalleryState =>
  state.retryAt === null ? state : { ...state, retryAt: null };

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
export function shouldFetch(state: GalleryState, now: number = Date.now()): boolean {
  if (!state.open || state.loading) return false;
  if (state.retryAt !== null && now < state.retryAt) return false;
  if (!state.loaded) return true;
  // Against the announcement the last listing answered, not against the
  // entries it held: an announced entry the store has already evicted never
  // appears in any listing, and comparing entries refetched it forever.
  return state.latestN !== null && (state.satisfiedN === null || state.latestN > state.satisfiedN);
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
/**
 * The whole room as one file: `zip` is every stored frame plus a manifest,
 * `avi` is a Motion JPEG video of them, drawing on the left and result on the
 * right. Plain links rather than fetches - the server streams a temp file it
 * deletes afterwards, and the browser's own download does that better than we
 * could with a blob.
 */
export const historyExportUrl = (roomId: string, kind: 'zip' | 'avi'): string =>
  `/rooms/${encodeURIComponent(roomId)}/history.${kind}`;

/** The same name the server's Content-Disposition asks for. */
export const historyExportFileName = (roomId: string, kind: 'zip' | 'avi'): string =>
  `brushjam-${roomId}-history.${kind}`;

export const formatLatency = (ms: number): string =>
  !Number.isFinite(ms) || ms < 0 ? '' : ms >= 10_000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`;

export function formatTime(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '';
  const date = new Date(ms);
  const two = (n: number): string => String(n).padStart(2, '0');
  return `${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`;
}
