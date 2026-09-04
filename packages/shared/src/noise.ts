/**
 * Deterministic pixel noise for the noise pen.
 *
 * The value of a pixel depends only on the stroke id and the pixel's *world*
 * coordinates, so a server-side crop (rendered with the crop origin subtracted)
 * produces exactly the same pixels as the client's full-size layer. Nothing
 * here is random: two renders of the same stroke are byte-identical.
 */

/** 32-bit FNV-1a of a string, used to turn a stroke id into a seed. */
export function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i) & 0xff;
    // hash *= 16777619, in 32-bit arithmetic
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** One 32-bit hash of (seed, x, y); the three low bytes become RGB. */
export function noiseHash(seed: number, x: number, y: number): number {
  let h = (seed ^ Math.imul(x | 0, 0x9e3779b1) ^ Math.imul(y | 0, 0x85ebca77)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  return (h ^ (h >>> 15)) >>> 0;
}

/** Uniform RGB noise for one world pixel. */
export function noiseRGB(seed: number, x: number, y: number): [number, number, number] {
  const h = noiseHash(seed, x, y);
  return [h & 0xff, (h >>> 8) & 0xff, (h >>> 16) & 0xff];
}
