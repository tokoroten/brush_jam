import { describe, expect, it, vi } from 'vitest';
import { copyText, type CopyDeps } from '../src/clipboard.js';

/**
 * Review 10 finding 1: the playtest URL is plain HTTP on a LAN IP, which is
 * precisely where navigator.clipboard does not exist. The button used to throw
 * into a floating promise and do nothing at all.
 */
describe('copyText', () => {
  it('uses the modern API when it is available', async () => {
    const writeText = vi.fn(async () => {});
    const legacyCopy = vi.fn(() => true);
    await expect(copyText('http://192.168.0.66:5173/r/abc', { clipboard: { writeText }, legacyCopy })).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith('http://192.168.0.66:5173/r/abc');
    expect(legacyCopy).not.toHaveBeenCalled();
  });

  it('falls back to execCommand when there is no clipboard API', async () => {
    const legacyCopy = vi.fn(() => true);
    await expect(copyText('http://192.168.0.66:5173/r/abc', { legacyCopy })).resolves.toBe(true);
    expect(legacyCopy).toHaveBeenCalledWith('http://192.168.0.66:5173/r/abc');
  });

  it('falls back when the clipboard API exists but rejects (insecure origin)', async () => {
    const writeText = vi.fn(async () => {
      throw new Error('Document is not focused / not a secure context');
    });
    const legacyCopy = vi.fn(() => true);
    await expect(copyText('u', { clipboard: { writeText }, legacyCopy })).resolves.toBe(true);
    expect(legacyCopy).toHaveBeenCalled();
  });

  it('reports failure rather than throwing when neither route works', async () => {
    const deps: CopyDeps = { legacyCopy: () => false };
    await expect(copyText('u', deps)).resolves.toBe(false);
  });

  it('reports failure when the legacy path throws too', async () => {
    await expect(
      copyText('u', {
        legacyCopy: () => {
          throw new Error('execCommand is not a function');
        },
      }),
    ).resolves.toBe(false);
  });

  it('reports failure when there is no route at all', async () => {
    await expect(copyText('u', {})).resolves.toBe(false);
  });
});
