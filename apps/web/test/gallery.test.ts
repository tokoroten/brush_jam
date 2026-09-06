import { describe, expect, it, vi } from 'vitest';
import type { HistoryEntry } from '@brushjam/shared';
import {
  GALLERY_RETRY_MAX_MS,
  GALLERY_RETRY_MIN_MS,
  applyHistorySettings,
  formatLatency,
  galleryAnnounced,
  galleryBackoffMs,
  galleryFailed,
  galleryLoaded,
  galleryLoading,
  galleryRetryDue,
  historyExportFileName,
  historyExportUrl,
  initialGallery,
  selectEntry,
  selectedEntry,
  shouldFetch,
  toggleGallery,
  type HistoryTarget,
} from '../src/gallery.js';
import type { SharedDraft } from '../src/sharedDraft.js';

const entry = (n: number, over: Partial<HistoryEntry> = {}): HistoryEntry => ({
  n,
  url: `/rooms/abcd/history/${n}.jpg`,
  time: 1_700_000_000_000 + n,
  aiRevision: n,
  aiGeneration: n + 1,
  prompt: `prompt ${n}`,
  negativePrompt: '',
  denoise: 0.8,
  seed: 1234,
  profile: 'fast',
  aiResolution: 768,
  latencyMs: 1800,
  ...over,
});

const listing = (entries: HistoryEntry[], enabled = true) => ({ roomId: 'abcd', enabled, entries });

describe('when the strip fetches', () => {
  it('never while it is closed', () => {
    expect(shouldFetch(initialGallery)).toBe(false);
    expect(shouldFetch(galleryAnnounced(initialGallery, 5))).toBe(false);
  });

  it('once on opening, and not again until something newer exists', () => {
    let state = toggleGallery(initialGallery);
    expect(shouldFetch(state)).toBe(true);
    state = galleryLoading(state);
    expect(shouldFetch(state)).toBe(false); // no second request in flight
    state = galleryLoaded(state, listing([entry(1), entry(0)]));
    expect(shouldFetch(state)).toBe(false);

    state = galleryAnnounced(state, 2);
    expect(shouldFetch(state)).toBe(true);
    state = galleryLoaded(galleryLoading(state), listing([entry(2), entry(1), entry(0)]));
    expect(shouldFetch(state)).toBe(false);
  });

  it('refetches an empty history once the room has generated something', () => {
    let state = galleryLoaded(toggleGallery(initialGallery), listing([]));
    expect(shouldFetch(state)).toBe(false);
    state = galleryAnnounced(state, 0);
    expect(shouldFetch(state)).toBe(true);
  });

  it('ignores an announcement that is not newer, and a missing one', () => {
    const state = galleryAnnounced(toggleGallery(initialGallery), 4);
    expect(galleryAnnounced(state, 3)).toBe(state);
    expect(galleryAnnounced(state, 4)).toBe(state);
    expect(galleryAnnounced(state, undefined)).toBe(state);
    expect(galleryAnnounced(state, null)).toBe(state);
    expect(galleryAnnounced(state, 5).latestN).toBe(5);
  });

  it('does not un-announce a result that arrived while a listing was in flight', () => {
    let state = galleryAnnounced(toggleGallery(initialGallery), 9);
    state = galleryLoaded(state, listing([entry(7)])); // the older answer
    expect(state.latestN).toBe(9);
    expect(shouldFetch(state)).toBe(true);
  });
});

describe('the listing', () => {
  it('is held newest first whatever order it arrives in', () => {
    const state = galleryLoaded(toggleGallery(initialGallery), listing([entry(0), entry(2), entry(1)]));
    expect(state.entries.map((e) => e.n)).toEqual([2, 1, 0]);
    expect(state.loaded).toBe(true);
    expect(state.loading).toBe(false);
  });

  it('drops a selection whose entry has been evicted', () => {
    let state = galleryLoaded(toggleGallery(initialGallery), listing([entry(1), entry(0)]));
    state = selectEntry(state, 0);
    expect(selectedEntry(state)?.n).toBe(0);
    state = galleryLoaded(state, listing([entry(2), entry(1)])); // 0 was evicted
    expect(state.selected).toBeNull();
    expect(selectedEntry(state)).toBeNull();
  });

  it('keeps a selection that is still there', () => {
    let state = galleryLoaded(toggleGallery(initialGallery), listing([entry(1), entry(0)]));
    state = selectEntry(state, 1);
    state = galleryLoaded(state, listing([entry(2), entry(1), entry(0)]));
    expect(selectedEntry(state)?.prompt).toBe('prompt 1');
  });

  it('remembers that the server keeps no history', () => {
    const state = galleryLoaded(toggleGallery(initialGallery), listing([], false));
    expect(state.enabled).toBe(false);
  });

  it('records a failure without losing what is on screen', () => {
    let state = galleryLoaded(toggleGallery(initialGallery), listing([entry(0)]));
    state = galleryFailed(galleryLoading(state), 'server said 500');
    expect(state.error).toBe('server said 500');
    expect(state.entries).toHaveLength(1);
    expect(state.loading).toBe(false);
  });

  it('closes the large view when the strip is closed', () => {
    let state = selectEntry(galleryLoaded(toggleGallery(initialGallery), listing([entry(0)])), 0);
    state = toggleGallery(state);
    expect(state.open).toBe(false);
    expect(state.selected).toBeNull();
  });

  it('clicking the open thumbnail again puts it away', () => {
    let state = galleryLoaded(toggleGallery(initialGallery), listing([entry(0)]));
    state = selectEntry(state, 0);
    state = selectEntry(state, 0);
    expect(state.selected).toBeNull();
  });
});

function draft<T>(value: T): SharedDraft<T> & { sent: T[] } {
  const sent: T[] = [];
  const state = { value, sent } as SharedDraft<T> & { sent: T[] };
  return Object.assign(state, {
    dirty: false,
    foreign: null,
    set: (next: T) => {
      state.value = next;
    },
    flush: () => sent.push(state.value),
    adopt: () => {},
    onFocus: () => {},
    onBlur: () => {},
  });
}

function target(maxDenoise = 0.95) {
  const prompt = draft('');
  const negative = draft('');
  const denoise = draft(0.5);
  const seed = draft(0);
  const profiles: HistoryTarget['profiles'] = ['fast', 'quality'];
  const send = vi.fn();
  const t: HistoryTarget = { prompt, negative, denoise, seed, maxDenoise, profiles, send };
  return { t, prompt, negative, denoise, seed, send };
}

describe('use these settings', () => {
  it('writes prompt, negative, denoise and seed, and sends each immediately', () => {
    const { t, prompt, negative, denoise, seed } = target();
    applyHistorySettings(entry(3, { prompt: 'a hill', negativePrompt: 'blurry', denoise: 0.65, seed: 99 }), t);
    expect(prompt.value).toBe('a hill');
    expect(prompt.sent).toEqual(['a hill']);
    expect(negative.value).toBe('blurry');
    expect(negative.sent).toEqual(['blurry']);
    expect(denoise.value).toBe(0.65);
    expect(seed.value).toBe(99);
    expect(seed.sent).toEqual([99]);
  });

  it('does not replay the profile, which is what this machine can do now', () => {
    const { t, send } = target();
    applyHistorySettings(entry(1, { profile: 'quality' }), t);
    expect(send).not.toHaveBeenCalled();
  });

  it('clamps a denoise the room can no longer express', () => {
    const { t, denoise } = target(0.8);
    applyHistorySettings(entry(1, { denoise: 0.95 }), t);
    expect(denoise.value).toBe(0.8);
  });

  it('survives an entry an older server wrote without every field', () => {
    const { t, prompt, negative, seed } = target();
    applyHistorySettings({ n: 1, url: '/x' } as unknown as HistoryEntry, t);
    expect(prompt.value).toBe('');
    expect(negative.value).toBe('');
    expect(seed.sent).toEqual([]);
  });
});

describe('the whole-room exports', () => {
  it('addresses the two server routes, and names the file the way the server does', () => {
    expect(historyExportUrl('abcd', 'zip')).toBe('/rooms/abcd/history.zip');
    expect(historyExportUrl('abcd', 'avi')).toBe('/rooms/abcd/history.avi');
    expect(historyExportFileName('abcd', 'zip')).toBe('brushjam-abcd-history.zip');
    expect(historyExportFileName('abcd', 'avi')).toBe('brushjam-abcd-history.avi');
  });

  it('never lets a room id out of its path segment', () => {
    expect(historyExportUrl('../secret', 'zip')).toBe('/rooms/..%2Fsecret/history.zip');
  });
});

describe('formatting', () => {
  it('reads like the status pill', () => {
    expect(formatLatency(1800)).toBe('1.8 s');
    expect(formatLatency(12_400)).toBe('12 s');
    expect(formatLatency(-1)).toBe('');
  });
});


/**
 * Review 3 finding 3: a failure only cleared `loading`, so `shouldFetch` was
 * true again immediately and the effect started another request in the same
 * tick - a server that was down got a continuous stream of them. And an
 * announced entry the store had already evicted never appeared in any
 * listing, so comparing `latestN` against the entries refetched forever.
 */
describe('when fetching goes wrong', () => {
  it('waits, doubling, instead of retrying at once', () => {
    let state = galleryLoading(toggleGallery(initialGallery));
    state = galleryFailed(state, 'server said 503', 1_000);
    expect(state.error).toBe('server said 503');
    expect(shouldFetch(state, 1_000)).toBe(false);
    expect(shouldFetch(state, 1_999)).toBe(false);
    expect(shouldFetch(state, 2_000)).toBe(true); // a second later

    state = galleryFailed(galleryLoading(state), 'server said 503', 2_000);
    expect(shouldFetch(state, 3_999)).toBe(false);
    expect(shouldFetch(state, 4_000)).toBe(true); // then two

    state = galleryFailed(galleryLoading(state), 'server said 503', 4_000);
    expect(shouldFetch(state, 8_000)).toBe(true); // then four
  });

  it('stops doubling at half a minute', () => {
    expect(galleryBackoffMs(1)).toBe(GALLERY_RETRY_MIN_MS);
    expect(galleryBackoffMs(4)).toBe(8 * GALLERY_RETRY_MIN_MS);
    expect(galleryBackoffMs(6)).toBe(GALLERY_RETRY_MAX_MS); // capped, not 32 s
    expect(galleryBackoffMs(20)).toBe(GALLERY_RETRY_MAX_MS);
  });

  it('forgets the backoff as soon as a listing arrives', () => {
    let state = galleryFailed(galleryLoading(toggleGallery(initialGallery)), 'boom', 1_000);
    expect(state.failures).toBe(1);
    state = galleryLoaded(galleryRetryDue(state), listing([entry(0)]));
    expect(state.failures).toBe(0);
    expect(state.retryAt).toBeNull();
    expect(state.error).toBeNull();
    // ...and the next failure starts from a second again.
    state = galleryFailed(galleryLoading(state), 'boom', 5_000);
    expect(state.retryAt).toBe(5_000 + GALLERY_RETRY_MIN_MS);
  });

  it('does not loop when the announced entry has already been evicted', () => {
    // The room announces result 7; by the time the strip asks, the store has
    // evicted it and the listing comes back without it (here: empty).
    let state = galleryAnnounced(toggleGallery(initialGallery), 7);
    state = galleryLoaded(galleryLoading(state), listing([]));
    expect(state.latestN).toBe(7);
    expect(shouldFetch(state, 0)).toBe(false);
    // A genuinely newer result still refetches.
    state = galleryAnnounced(state, 8);
    expect(shouldFetch(state, 0)).toBe(true);
  });

  it('still fetches for an announcement that arrived mid-request', () => {
    let state = galleryAnnounced(toggleGallery(initialGallery), 3);
    state = galleryLoading(state);
    state = galleryAnnounced(state, 4); // lands while the request is in flight
    state = galleryLoaded(state, listing([entry(3), entry(2)]));
    expect(shouldFetch(state, 0)).toBe(true);
  });
});
