/**
 * The AI overlay: the generated picture laid over the drawing, to trace on.
 *
 * It is a sheet of tracing paper the other way round - the machine's version
 * on top of yours, at whatever opacity lets you see both. Everything here is
 * display state and lives only in this browser: no message, no protocol field,
 * nothing anybody else in the room can see. That is deliberate, and it is why
 * the overlay may be pinned to an old result while the room has moved on.
 *
 * The rules that are easy to get wrong, and are therefore tested here rather
 * than in the markup:
 *
 * - **Pinned or following.** With nothing pinned the overlay shows the room's
 *   latest result and changes under you as the AI works. Pinning freezes one
 *   image - the one on screen, or one from the history strip - so a trace does
 *   not shift halfway through.
 * - **Peeking.** Tab hides the overlay for as long as it is held, which is the
 *   only way to see what you have actually drawn. Held, not toggled: a toggle
 *   would leave the overlay off after a glance and the next stroke would go
 *   down blind.
 * - **The URL is derived, never stored.** A pinned entry keeps its address; a
 *   followed one is rebuilt from the revision, so the browser caches each
 *   result and refetches exactly when there is a new one.
 */

import type { StorageLike } from './session.js';

export interface OverlayState {
  /** The player has turned it on. */
  on: boolean;
  /** 0-1. 0.4 is where lines stay readable through the picture. */
  opacity: number;
  /** A frozen image, or null to follow the room's latest result. */
  pinned: string | null;
  /** Tab is down: hidden until it comes back up. Never persisted. */
  peeking: boolean;
}

export const DEFAULT_OVERLAY_OPACITY = 0.4;

export const initialOverlay: OverlayState = {
  on: false,
  opacity: DEFAULT_OVERLAY_OPACITY,
  pinned: null,
  peeking: false,
};

export const overlayKey = (roomId: string): string => `brushjam.overlay.${roomId}`;

const clampOpacity = (value: unknown): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DEFAULT_OVERLAY_OPACITY;
  return Math.min(1, Math.max(0, value));
};

/** The AI result the room is showing now. Keyed by revision, so it caches. */
export const aiOverlayUrl = (roomId: string, aiRevision: number): string | null =>
  aiRevision > 0 ? `/rooms/${roomId}/ai.png?v=${aiRevision}` : null;

/** One entry from the history strip (the same URL the gallery thumbnails use). */
export const historyOverlayUrl = (roomId: string, n: number): string =>
  `/rooms/${roomId}/history/${n}.jpg`;

/** What to draw: the pinned image, or the room's latest, or nothing yet. */
export function overlaySource(state: OverlayState, roomId: string, aiRevision: number): string | null {
  return state.pinned ?? aiOverlayUrl(roomId, aiRevision);
}

/** Whether the stage should draw it at all this frame. */
export const overlayVisible = (state: OverlayState, source: string | null): boolean =>
  state.on && !state.peeking && source !== null;

export const toggleOverlay = (state: OverlayState): OverlayState => ({ ...state, on: !state.on });

export const setOverlayOpacity = (state: OverlayState, opacity: number): OverlayState => ({
  ...state,
  opacity: clampOpacity(opacity),
});

/**
 * Freeze what is on screen.
 *
 * `source` is what `overlaySource` returns right now, which is why pinning
 * before the first generation does nothing: there is no picture to freeze, and
 * pinning "nothing" would leave a pin nobody could clear by looking at it.
 */
export function pinOverlay(state: OverlayState, source: string | null): OverlayState {
  return source === null ? state : { ...state, pinned: source };
}

/** Follow the room's latest result again. */
export const unpinOverlay = (state: OverlayState): OverlayState => ({ ...state, pinned: null });

/**
 * The gallery's "pin as overlay": that entry, and the overlay turned on.
 *
 * Turning it on is the point of the button - the entry was chosen from a list
 * of pictures, so the answer to "what happens now" has to be that this picture
 * appears on the canvas.
 */
export const pinAsOverlay = (state: OverlayState, source: string): OverlayState => ({
  ...state,
  pinned: source,
  on: true,
});

/** Tab went down or came up. */
export const peekOverlay = (state: OverlayState, held: boolean): OverlayState =>
  state.peeking === held ? state : { ...state, peeking: held };

/**
 * Whether Tab belongs to the overlay in this moment.
 *
 * Not while a text field has the focus: Tab is how you leave one, and stealing
 * it there would trap the keyboard in the prompt box. Not with a modifier
 * either - Ctrl+Tab and Alt+Tab are the browser's and the desktop's.
 */
export function overlayTakesTab(
  event: { key: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean },
  /** `KeyboardEvent.target`, duck-typed: this runs against jsdom, a real DOM
   *  and a plain object in the tests, and only ever asks two questions of it. */
  target: unknown,
): boolean {
  if (event.key !== 'Tab' || event.ctrlKey || event.metaKey || event.altKey) return false;
  return !isTyping(target);
}

function isTyping(target: unknown): boolean {
  const element = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (!element || typeof element !== 'object') return false;
  if (element.isContentEditable === true) return true;
  const tag = typeof element.tagName === 'string' ? element.tagName.toUpperCase() : '';
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

/** localStorage when the browser allows it; undefined in tests and SSR. */
export function overlayStorage(): StorageLike | undefined {
  try {
    return typeof localStorage === 'undefined' ? undefined : localStorage;
  } catch {
    return undefined;
  }
}

/**
 * What was remembered for this room, or the defaults.
 *
 * Per room, because a pin is a picture from *that* room and would be nonsense
 * anywhere else. Anything unreadable, malformed or written by an older build
 * falls back rather than failing: an overlay setting must never stop a room
 * from opening.
 */
export function loadOverlay(storage: StorageLike | undefined, roomId: string): OverlayState {
  try {
    const raw = storage?.getItem(overlayKey(roomId));
    if (!raw) return initialOverlay;
    const parsed = JSON.parse(raw) as Partial<OverlayState> | null;
    if (typeof parsed !== 'object' || parsed === null) return initialOverlay;
    return {
      on: parsed.on === true,
      opacity: clampOpacity(parsed.opacity),
      pinned: typeof parsed.pinned === 'string' && parsed.pinned !== '' ? parsed.pinned : null,
      // A key cannot be held across a page load.
      peeking: false,
    };
  } catch {
    return initialOverlay;
  }
}

/** Best effort: a full or blocked quota must not break drawing. */
export function saveOverlay(
  storage: StorageLike | undefined,
  roomId: string,
  state: OverlayState,
): void {
  try {
    storage?.setItem(
      overlayKey(roomId),
      JSON.stringify({ on: state.on, opacity: state.opacity, pinned: state.pinned }),
    );
  } catch {
    /* the overlay is still correct in memory for this session */
  }
}
