import { describe, expect, it } from 'vitest';
import { newId } from '../src/id.js';

describe('newId', () => {
  it('produces url-safe ids of the requested length', () => {
    const id = newId(16);
    expect(id).toHaveLength(16);
    expect(id).toMatch(/^[a-z0-9]+$/);
  });

  it('does not collide over many draws', () => {
    const seen = new Set(Array.from({ length: 2000 }, () => newId()));
    expect(seen.size).toBe(2000);
  });

  it('works without a Web Crypto implementation (insecure http contexts)', () => {
    const id = newId(12, undefined);
    expect(id).toHaveLength(12);
    expect(id).toMatch(/^[a-z0-9]+$/);
  });
});
