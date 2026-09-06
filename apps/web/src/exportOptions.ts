/**
 * Taking something home: which of the four ways out are available, and where
 * they point.
 *
 * The header used to carry a button per picture ("save AI", "save drawing"),
 * and the history strip grew two more links when the server learned to build a
 * zip and a video. That is four buttons in two places for one intention, in a
 * header bar that had already run out of room, so they are one "export" button
 * and a dialog now - and this module is the part of it worth testing: what is
 * offered, what is greyed out and why, and the address behind each link.
 *
 * The two PNGs are always offered. They are made from what this browser
 * already has - the AI raster the server holds, and the layers composited here
 * (save.ts) - so neither depends on the server keeping a history.
 */

import { historyExportUrl } from './gallery.js';

/** The frame rates offered. Fewer, larger steps than a slider: this is "how
 *  fast should the flip-book go", not a setting anybody wants to tune. */
export const EXPORT_FPS_CHOICES = [2, 4, 8] as const;

/** The server's own default (`DEFAULT_FPS` in brushjam/export.py). */
export const DEFAULT_EXPORT_FPS = 4;

export const EXPORT_ZIP_HINT =
  'every frame this room made - draw_NNNNN.jpg, gen_NNNNN.jpg - plus a manifest.json of the settings behind each one';

export const EXPORT_VIDEO_HINT =
  'left: your drawing, right: the AI result, one frame per generation; plays in VLC / video editors';

export interface HistoryExportAvailability {
  /** There is something to export. */
  ready: boolean;
  /** Why not, in the words the history strip already uses. Null when ready. */
  hint: string | null;
}

/**
 * Whether the zip and the video are worth offering.
 *
 * Two different noes: a server started with `HISTORY_ENABLED=0` will never
 * have anything, and a room that has not generated yet will. Saying which is
 * the difference between "this build cannot" and "not yet", and a disabled
 * link with no reason beside it is the thing people file bugs about.
 *
 * `latestN` counts as well as the listed entries, because the strip is only
 * fetched while it is open: a room can have announced a dozen results through
 * `ai_result` without anybody having looked at the history once.
 */
export function historyExportAvailability(input: {
  /** False once a listing has said the server keeps no history. */
  enabled: boolean;
  /** How many entries the strip has actually listed. */
  entries: number;
  /** The newest entry the room has announced, listed or not. */
  latestN: number | null;
}): HistoryExportAvailability {
  if (!input.enabled) {
    return { ready: false, hint: 'this server keeps no history (HISTORY_ENABLED=0)' };
  }
  if (input.entries === 0 && input.latestN === null) {
    return { ready: false, hint: 'nothing generated in this room yet' };
  }
  return { ready: true, hint: null };
}

/** Anything that is not one of the offered rates is the default, not an error. */
export const clampExportFps = (fps: number): number =>
  (EXPORT_FPS_CHOICES as readonly number[]).includes(fps) ? fps : DEFAULT_EXPORT_FPS;

/**
 * The video, at a chosen rate.
 *
 * The rate is a query parameter rather than a form: the link has to be a plain
 * `<a download>` so the browser saves the stream itself, which it does far
 * better than a fetch into a blob would (the server streams a temp file and
 * deletes it afterwards, and a room's history can be hundreds of megabytes).
 */
export const historyVideoUrl = (roomId: string, fps: number): string =>
  `${historyExportUrl(roomId, 'avi')}?fps=${clampExportFps(fps)}`;

/** The zip needs no options at all. */
export const historyZipUrl = (roomId: string): string => historyExportUrl(roomId, 'zip');
