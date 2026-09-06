/**
 * Saving a picture out of the room.
 *
 * Two buttons, two different things. "save AI" downloads what the server made
 * - it is fetched rather than read off the AI canvas, because that canvas is a
 * view assembled from patches and a browser will not let a tainted one be
 * exported anyway. "save drawing" is composited here from the layers, because
 * the server has no PNG of the human canvas: it renders one only as model
 * input, at generation size, and never keeps it.
 *
 * The file name is what somebody finds in their downloads folder a week later,
 * so it carries the room and the revision it came from.
 */

import { ctxOf, drawHumanFrame, scratchCanvas, type HumanFrameModel } from './raster.js';

/** Anything outside this would be a path, a shell character or a surprise. */
const UNSAFE = /[^a-z0-9-]+/gi;

function slug(value: string, fallback: string): string {
  const cleaned = (value ?? '').replace(UNSAFE, '').slice(0, 32);
  return cleaned || fallback;
}

function counter(value: number): number {
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
}

/**
 * `brushjam-<room>-<n>-<what>.<ext>`.
 *
 * The number is the revision or entry the picture came from, so two saves from
 * the same room sort in the order they were made and never collide - which is
 * what the browser's "(1)" suffix would otherwise be doing for you, silently.
 */
export function saveFileName(
  roomId: string,
  kind: 'ai' | 'drawing',
  n: number,
  extension = 'png',
): string {
  return `brushjam-${slug(roomId, 'room')}-${counter(n)}-${kind}.${extension}`;
}

export const aiFileName = (roomId: string, aiRevision: number): string =>
  saveFileName(roomId, 'ai', aiRevision, 'png');

export const drawingFileName = (roomId: string, humanRevision: number): string =>
  saveFileName(roomId, 'drawing', humanRevision, 'png');

/** A saved gallery entry, which is a JPEG and numbered by the store. */
export const historyFileName = (roomId: string, n: number): string =>
  saveFileName(roomId, 'ai', n, 'jpg');

export interface DownloadDeps {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
  /** Given a URL and a name, make the browser save it. */
  click(url: string, filename: string): void;
  fetch(url: string): Promise<{ ok: boolean; status: number; blob(): Promise<Blob> }>;
}

export function browserDownloadDeps(): DownloadDeps {
  return {
    createObjectURL: (blob) => URL.createObjectURL(blob),
    revokeObjectURL: (url) => URL.revokeObjectURL(url),
    click: (url, filename) => {
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      // Firefox only follows a click on an anchor that is in the document.
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    },
    fetch: (url) => fetch(url),
  };
}

export function downloadBlob(blob: Blob, filename: string, deps: DownloadDeps): void {
  const url = deps.createObjectURL(blob);
  try {
    deps.click(url, filename);
  } finally {
    // Not immediately: Safari has not started the download when click returns.
    setTimeout(() => deps.revokeObjectURL(url), 10_000);
  }
}

/** Fetch a server-side picture and save it. Returns false if it is not there. */
export async function downloadUrl(url: string, filename: string, deps: DownloadDeps): Promise<boolean> {
  const response = await deps.fetch(url);
  if (!response.ok) return false;
  downloadBlob(await response.blob(), filename, deps);
  return true;
}

/** "save AI": the room's current AI raster, as the server holds it. */
export function saveAiImage(
  roomId: string,
  aiRevision: number,
  deps: DownloadDeps,
): Promise<boolean> {
  // Cache-busted: the URL is stable while the picture behind it is not.
  return downloadUrl(
    `/rooms/${roomId}/ai.png?v=${aiRevision}`,
    aiFileName(roomId, aiRevision),
    deps,
  );
}

export interface CanvasBlobs {
  toBlob(canvas: HTMLCanvasElement): Promise<Blob | null>;
}

export const browserCanvasBlobs = (): CanvasBlobs => ({
  toBlob: (canvas) =>
    new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/png');
    }),
});

/**
 * "save drawing": every visible layer, at canvas size, on white.
 *
 * `drawHumanFrame` is the renderer the stage uses, so what is saved is exactly
 * what is on screen - including reference images and in-progress strokes -
 * rather than a second implementation that will drift from it.
 */
export async function composeDrawing(model: HumanFrameModel, blobs: CanvasBlobs): Promise<Blob | null> {
  // Through the same seam the renderer uses, so a test can supply a canvas.
  const canvas = scratchCanvas(model.canvasSize, model.canvasSize);
  drawHumanFrame(ctxOf(canvas), model);
  return blobs.toBlob(canvas);
}

export async function saveDrawing(
  roomId: string,
  humanRevision: number,
  model: HumanFrameModel,
  deps: DownloadDeps,
  blobs: CanvasBlobs = browserCanvasBlobs(),
): Promise<boolean> {
  const blob = await composeDrawing(model, blobs);
  if (!blob) return false;
  downloadBlob(blob, drawingFileName(roomId, humanRevision), deps);
  return true;
}
