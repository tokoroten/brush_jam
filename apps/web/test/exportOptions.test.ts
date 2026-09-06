import { describe, expect, it } from 'vitest';
import {
  DEFAULT_EXPORT_FPS,
  EXPORT_FPS_CHOICES,
  clampExportFps,
  historyExportAvailability,
  historyVideoUrl,
  historyZipUrl,
} from '../src/exportOptions.js';
import { historyExportFileName } from '../src/gallery.js';

const ROOM = 'abcd';

describe('what the dialog offers', () => {
  it('offers the history once the room has generated something', () => {
    expect(historyExportAvailability({ enabled: true, entries: 3, latestN: 12 })).toEqual({
      ready: true,
      hint: null,
    });
  });

  it('counts an announcement nobody has opened the strip to see', () => {
    // The listing is only fetched while the strip is open, so a room can have
    // a dozen results and an empty `entries`.
    expect(historyExportAvailability({ enabled: true, entries: 0, latestN: 4 }).ready).toBe(true);
  });

  it('says which kind of no it is', () => {
    // "cannot" ...
    const off = historyExportAvailability({ enabled: false, entries: 0, latestN: null });
    expect(off.ready).toBe(false);
    expect(off.hint).toContain('HISTORY_ENABLED=0');
    // ...against "not yet", which is a different thing to tell somebody.
    const unknown = historyExportAvailability({ enabled: true, entries: 0, latestN: null, loaded: false });
    expect(unknown.ready).toBe(false);
    expect(unknown.hint).toMatch(/checking/);
    const empty = historyExportAvailability({ enabled: true, entries: 0, latestN: null, loaded: true });
    expect(empty.ready).toBe(false);
    expect(empty.hint).toBe('nothing generated in this room yet');
    // A server with the history off never announces anything, but if one ever
    // did, off wins: there is nothing to fetch.
    expect(historyExportAvailability({ enabled: false, entries: 2, latestN: 9 }).ready).toBe(false);
  });
});

describe('where the links point', () => {
  it('asks for the video at the chosen rate', () => {
    expect(historyVideoUrl(ROOM, 2)).toBe('/rooms/abcd/history.avi?fps=2');
    expect(historyVideoUrl(ROOM, 8)).toBe('/rooms/abcd/history.avi?fps=8');
    expect(historyZipUrl(ROOM)).toBe('/rooms/abcd/history.zip');
  });

  it("offers a few rates and defaults to the server's own", () => {
    expect([...EXPORT_FPS_CHOICES]).toEqual([2, 4, 8]);
    expect(EXPORT_FPS_CHOICES).toContain(DEFAULT_EXPORT_FPS);
    expect(DEFAULT_EXPORT_FPS).toBe(4);
  });

  it('falls back to the default rather than asking for a rate nobody offered', () => {
    expect(clampExportFps(5)).toBe(DEFAULT_EXPORT_FPS);
    expect(clampExportFps(Number.NaN)).toBe(DEFAULT_EXPORT_FPS);
    expect(clampExportFps(-1)).toBe(DEFAULT_EXPORT_FPS);
    expect(historyVideoUrl(ROOM, 999)).toBe(`/rooms/abcd/history.avi?fps=${DEFAULT_EXPORT_FPS}`);
  });

  it('escapes a room id that would otherwise reshape the path', () => {
    expect(historyVideoUrl('a/b', 4)).toBe('/rooms/a%2Fb/history.avi?fps=4');
  });

  it('keeps the file names the server asks for', () => {
    expect(historyExportFileName(ROOM, 'zip')).toBe('brushjam-abcd-history.zip');
    expect(historyExportFileName(ROOM, 'avi')).toBe('brushjam-abcd-history.avi');
  });
});
