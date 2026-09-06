import { createCanvas } from '@napi-rs/canvas';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { Layer } from '@brushjam/shared';
import { setScratchCanvasFactory, type HumanFrameModel } from '../src/raster.js';
import {
  aiFileName,
  composeDrawing,
  downloadBlob,
  drawingFileName,
  historyFileName,
  saveAiImage,
  saveDrawing,
  saveFileName,
  type DownloadDeps,
} from '../src/save.js';

beforeAll(() => {
  setScratchCanvasFactory((w, h) => createCanvas(w, h) as unknown as HTMLCanvasElement);
});

function deps(overrides: Partial<DownloadDeps> = {}): DownloadDeps & { saved: string[] } {
  const saved: string[] = [];
  return {
    saved,
    createObjectURL: () => 'blob:test',
    revokeObjectURL: () => {},
    click: (_url, filename) => saved.push(filename),
    fetch: async () => ({ ok: true, status: 200, blob: async () => new Blob(['x']) }),
    ...overrides,
  };
}

describe('save file names', () => {
  it('carries the room and the revision it came from', () => {
    expect(aiFileName('abc123', 42)).toBe('brushjam-abc123-42-ai.png');
    expect(drawingFileName('abc123', 7)).toBe('brushjam-abc123-7-drawing.png');
    expect(historyFileName('abc123', 3)).toBe('brushjam-abc123-3-ai.jpg');
  });

  it('never builds a path or a surprise out of the room id', () => {
    // A room id is [a-z0-9]{4,16} server-side, but the client is handed
    // whatever is in the URL bar, and this becomes a file name.
    expect(saveFileName('../../etc/passwd', 'ai', 1)).toBe('brushjam-etcpasswd-1-ai.png');
    expect(saveFileName('a b"c', 'ai', 1)).toBe('brushjam-abc-1-ai.png');
    expect(saveFileName('', 'ai', 1)).toBe('brushjam-room-1-ai.png');
    expect(saveFileName('!!!', 'drawing', 1)).toBe('brushjam-room-1-drawing.png');
  });

  it('keeps the number a plain non-negative integer', () => {
    expect(aiFileName('abcd', 0)).toBe('brushjam-abcd-0-ai.png');
    expect(aiFileName('abcd', 3.7)).toBe('brushjam-abcd-3-ai.png');
    expect(aiFileName('abcd', -1)).toBe('brushjam-abcd-0-ai.png');
    expect(aiFileName('abcd', Number.NaN)).toBe('brushjam-abcd-0-ai.png');
  });
});

describe('downloading', () => {
  it('revokes the object URL, but not before the browser has started', () => {
    vi.useFakeTimers();
    const revokeObjectURL = vi.fn();
    const d = deps({ revokeObjectURL });
    downloadBlob(new Blob(['x']), 'a.png', d);
    expect(d.saved).toEqual(['a.png']);
    expect(revokeObjectURL).not.toHaveBeenCalled();
    vi.advanceTimersByTime(10_000);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:test');
    vi.useRealTimers();
  });

  it('saves the AI result the server holds, cache-busted by revision', async () => {
    const urls: string[] = [];
    const d = deps({
      fetch: async (url) => {
        urls.push(url);
        return { ok: true, status: 200, blob: async () => new Blob(['png']) };
      },
    });
    await expect(saveAiImage('abcd', 12, d)).resolves.toBe(true);
    expect(urls).toEqual(['/rooms/abcd/ai.png?v=12']);
    expect(d.saved).toEqual(['brushjam-abcd-12-ai.png']);
  });

  it('reports rather than saves an empty file when there is no AI result yet', async () => {
    const d = deps({ fetch: async () => ({ ok: false, status: 404, blob: async () => new Blob([]) }) });
    await expect(saveAiImage('abcd', 0, d)).resolves.toBe(false);
    expect(d.saved).toEqual([]);
  });
});

describe('composing the drawing', () => {
  const layer = (id: string, order: number): Layer => ({
    id,
    name: id,
    kind: 'draw',
    visible: true,
    locked: false,
    opacity: 1,
    order,
    includeInAI: true,
  });

  function model(): HumanFrameModel {
    const raster = createCanvas(32, 32);
    const ctx = raster.getContext('2d');
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(0, 0, 16, 32);
    return {
      canvasSize: 32,
      orderedLayers: [layer('a', 0)],
      layerCanvases: new Map([['a', raster as unknown as HTMLCanvasElement]]),
      live: new Map(),
      previewRaster: () => null,
    };
  }

  it('draws the visible layers at canvas size, on white', async () => {
    let painted: HTMLCanvasElement | null = null;
    const blob = await composeDrawing(model(), {
      toBlob: async (canvas) => {
        painted = canvas;
        return new Blob(['png']);
      },
    });
    expect(blob).not.toBeNull();
    const canvas = painted as unknown as ReturnType<typeof createCanvas>;
    expect(canvas.width).toBe(32);
    const ctx = canvas.getContext('2d');
    // The layer, and beside it the white the transparent canvas becomes.
    expect([...ctx.getImageData(4, 4, 1, 1).data]).toEqual([255, 0, 0, 255]);
    expect([...ctx.getImageData(28, 4, 1, 1).data]).toEqual([255, 255, 255, 255]);
  });

  it('saves it under the human revision', async () => {
    const d = deps();
    await expect(
      saveDrawing('abcd', 9, model(), d, { toBlob: async () => new Blob(['png']) }),
    ).resolves.toBe(true);
    expect(d.saved).toEqual(['brushjam-abcd-9-drawing.png']);
  });

  it('reports failure rather than saving nothing when the canvas gives no blob', async () => {
    const d = deps();
    await expect(saveDrawing('abcd', 9, model(), d, { toBlob: async () => null })).resolves.toBe(false);
    expect(d.saved).toEqual([]);
  });
});
