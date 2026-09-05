import { MAX_SEED } from './constants.js';

/**
 * The room's sampling seed.
 *
 * Both servers used to draw a fresh seed for every generation, so adding one
 * stroke reshuffled the whole picture and nobody could tell what their own
 * change had done. The room owns one instead, and re-rolling it is a
 * deliberate act - the dice next to the field.
 */
export const clampSeed = (value: number): number => {
  const whole = Math.trunc(value);
  if (whole >= 0 && whole <= MAX_SEED) return whole;
  return ((whole % (MAX_SEED + 1)) + MAX_SEED + 1) % (MAX_SEED + 1);
};

export const isSeed = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= MAX_SEED;

export const randomSeed = (): number => Math.floor(Math.random() * (MAX_SEED + 1));
