import { describe, expect, it } from 'vitest';
import {
  DEFAULT_OVERLAY_OPACITY,
  aiOverlayUrl,
  historyOverlayUrl,
  initialOverlay,
  loadOverlay,
  overlayKey,
  overlaySource,
  overlayTakesTab,
  overlayVisible,
  peekOverlay,
  pinAsOverlay,
  pinOverlay,
  saveOverlay,
  setOverlayOpacity,
  toggleOverlay,
  unpinOverlay,
  type OverlayState,
} from '../src/overlay.js';
import type { StorageLike } from '../src/session.js';

const ROOM = 'abcd';

/** localStorage, or as much of it as this module uses. */
class FakeStorage implements StorageLike {
  readonly items = new Map<string, string>();
  getItem(key: string): string | null {
    return this.items.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.items.set(key, value);
  }
}

/** The kind that throws on every call: private mode, blocked site data. */
class HostileStorage implements StorageLike {
  getItem(): string {
    throw new Error('access denied');
  }
  setItem(): void {
    throw new Error('quota exceeded');
  }
}

const on = (extra: Partial<OverlayState> = {}): OverlayState => ({
  ...initialOverlay,
  on: true,
  ...extra,
});

describe('turning the overlay on and off', () => {
  it('starts off, at the opacity where both pictures are readable', () => {
    expect(initialOverlay.on).toBe(false);
    expect(initialOverlay.opacity).toBe(DEFAULT_OVERLAY_OPACITY);
    expect(DEFAULT_OVERLAY_OPACITY).toBe(0.4);
    expect(initialOverlay.pinned).toBeNull();
  });

  it('toggles', () => {
    const shown = toggleOverlay(initialOverlay);
    expect(shown.on).toBe(true);
    expect(toggleOverlay(shown).on).toBe(false);
    // ...and the toggle is the only thing it changes: an overlay turned off
    // and on again is the same overlay.
    expect(toggleOverlay(toggleOverlay(on({ opacity: 0.7, pinned: '/a.png' })))).toEqual(
      on({ opacity: 0.7, pinned: '/a.png' }),
    );
  });

  it('keeps the opacity a fraction between none and all of it', () => {
    expect(setOverlayOpacity(initialOverlay, 0.75).opacity).toBe(0.75);
    expect(setOverlayOpacity(initialOverlay, 2).opacity).toBe(1);
    expect(setOverlayOpacity(initialOverlay, -1).opacity).toBe(0);
    expect(setOverlayOpacity(initialOverlay, Number.NaN).opacity).toBe(DEFAULT_OVERLAY_OPACITY);
    expect(setOverlayOpacity(initialOverlay, 0).opacity).toBe(0); // invisible, still on
  });
});

describe('which picture is shown', () => {
  it('follows the latest result in the room when nothing is pinned', () => {
    expect(overlaySource(on(), ROOM, 7)).toBe('/rooms/abcd/ai.png?v=7');
    // Keyed by revision rather than by the clock, so the browser caches one
    // result and refetches exactly when there is a new one.
    expect(overlaySource(on(), ROOM, 8)).toBe('/rooms/abcd/ai.png?v=8');
  });

  it('has nothing to show before the room has generated anything', () => {
    expect(aiOverlayUrl(ROOM, 0)).toBeNull();
    expect(overlaySource(on(), ROOM, 0)).toBeNull();
    expect(overlayVisible(on(), null)).toBe(false);
  });

  it('freezes what is on screen when it is pinned', () => {
    const pinned = pinOverlay(on(), overlaySource(on(), ROOM, 7));
    expect(pinned.pinned).toBe('/rooms/abcd/ai.png?v=7');
    // The room moves on; the overlay does not.
    expect(overlaySource(pinned, ROOM, 9)).toBe('/rooms/abcd/ai.png?v=7');
    expect(overlaySource(unpinOverlay(pinned), ROOM, 9)).toBe('/rooms/abcd/ai.png?v=9');
  });

  it('does not pin nothing', () => {
    // Before the first generation there is no picture to freeze, and a pin
    // nobody can see is a pin nobody can clear.
    const state = on();
    expect(pinOverlay(state, overlaySource(state, ROOM, 0))).toBe(state);
  });

  it('pins one entry from the history strip, and turns itself on', () => {
    expect(historyOverlayUrl(ROOM, 12)).toBe('/rooms/abcd/history/12.jpg');
    const state = pinAsOverlay(initialOverlay, historyOverlayUrl(ROOM, 12));
    expect(state.on).toBe(true); // the button's whole point
    expect(overlaySource(state, ROOM, 99)).toBe('/rooms/abcd/history/12.jpg');
  });

  it('is drawn only when it is on, not peeking, and has something to draw', () => {
    const url = '/rooms/abcd/ai.png?v=3';
    expect(overlayVisible(on(), url)).toBe(true);
    expect(overlayVisible(initialOverlay, url)).toBe(false);
    expect(overlayVisible(on({ peeking: true }), url)).toBe(false);
    expect(overlayVisible(on(), null)).toBe(false);
  });
});

describe('holding Tab to look underneath', () => {
  it('hides while it is held and comes back when it is let go', () => {
    const held = peekOverlay(on(), true);
    expect(overlayVisible(held, '/a.png')).toBe(false);
    expect(held.on).toBe(true); // hidden, not turned off
    expect(overlayVisible(peekOverlay(held, false), '/a.png')).toBe(true);
  });

  it('returns the same object when nothing changes, so no frame is wasted', () => {
    const state = on();
    expect(peekOverlay(state, false)).toBe(state);
    const held = peekOverlay(state, true);
    expect(peekOverlay(held, true)).toBe(held);
  });

  it('leaves Tab alone in a text field, where it is how you get out', () => {
    const tab = { key: 'Tab' };
    expect(overlayTakesTab(tab, null)).toBe(true);
    expect(overlayTakesTab(tab, { tagName: 'CANVAS' })).toBe(true);
    expect(overlayTakesTab(tab, { tagName: 'INPUT' })).toBe(false);
    expect(overlayTakesTab(tab, { tagName: 'textarea' })).toBe(false);
    expect(overlayTakesTab(tab, { tagName: 'SELECT' })).toBe(false);
    expect(overlayTakesTab(tab, { tagName: 'DIV', isContentEditable: true })).toBe(false);
  });

  it('leaves the browser and the desktop their own shortcuts', () => {
    expect(overlayTakesTab({ key: 'Tab', ctrlKey: true }, null)).toBe(false);
    expect(overlayTakesTab({ key: 'Tab', metaKey: true }, null)).toBe(false);
    expect(overlayTakesTab({ key: 'Tab', altKey: true }, null)).toBe(false);
    expect(overlayTakesTab({ key: 'a' }, null)).toBe(false);
  });
});

describe('what is remembered', () => {
  it('comes back after a reload, per room', () => {
    const storage = new FakeStorage();
    const state = on({ opacity: 0.65, pinned: '/rooms/abcd/history/4.jpg' });
    saveOverlay(storage, ROOM, state);
    expect(loadOverlay(storage, ROOM)).toEqual({ ...state, peeking: false });
    // A pin is a picture from *that* room, so another room starts fresh.
    expect(loadOverlay(storage, 'efgh')).toEqual(initialOverlay);
    expect(storage.items.has(overlayKey(ROOM))).toBe(true);
  });

  it('never comes back mid-peek', () => {
    const storage = new FakeStorage();
    saveOverlay(storage, ROOM, on({ peeking: true }));
    expect(loadOverlay(storage, ROOM).peeking).toBe(false);
    expect(storage.getItem(overlayKey(ROOM))).not.toContain('peeking');
  });

  it('falls back rather than failing on anything it cannot read', () => {
    const storage = new FakeStorage();
    storage.setItem(overlayKey(ROOM), 'not json at all');
    expect(loadOverlay(storage, ROOM)).toEqual(initialOverlay);

    storage.setItem(overlayKey(ROOM), 'null');
    expect(loadOverlay(storage, ROOM)).toEqual(initialOverlay);

    // A value from another build, or a hand-edited one.
    storage.setItem(overlayKey(ROOM), JSON.stringify({ on: 'yes', opacity: 'lots', pinned: 42 }));
    expect(loadOverlay(storage, ROOM)).toEqual(initialOverlay);

    storage.setItem(overlayKey(ROOM), JSON.stringify({ on: true, opacity: 5, pinned: '' }));
    expect(loadOverlay(storage, ROOM)).toEqual(on({ opacity: 1 }));
  });

  it('survives storage that is not there or throws', () => {
    expect(loadOverlay(undefined, ROOM)).toEqual(initialOverlay);
    expect(() => saveOverlay(undefined, ROOM, on())).not.toThrow();
    const hostile = new HostileStorage();
    expect(loadOverlay(hostile, ROOM)).toEqual(initialOverlay);
    expect(() => saveOverlay(hostile, ROOM, on())).not.toThrow();
  });
});
