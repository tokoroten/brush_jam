import { DIRTY_MERGE_PADDING } from './constants.js';
import { expandRect, rectsIntersect, unionRect, type Rect } from './geometry.js';

/**
 * Merge `next` into `regions`: any existing region whose bbox, expanded by
 * `padding`, intersects the (equally expanded) new region is absorbed. Repeats
 * until no further merges are possible, so chains collapse in one call.
 * Result order is stable, with the merged region appended last (= "most recent").
 */
export function mergeDirty(regions: readonly Rect[], next: Rect, padding = DIRTY_MERGE_PADDING): Rect[] {
  let merged = next;
  let kept = [...regions];
  let changed = true;
  while (changed) {
    changed = false;
    const rest: Rect[] = [];
    for (const r of kept) {
      if (rectsIntersect(expandRect(r, padding), expandRect(merged, padding))) {
        merged = unionRect(merged, r);
        changed = true;
      } else {
        rest.push(r);
      }
    }
    kept = rest;
  }
  kept.push(merged);
  return kept;
}

/** Merge a batch of rects into a region list, left to right. */
export function mergeDirtyAll(regions: readonly Rect[], nexts: readonly Rect[], padding = DIRTY_MERGE_PADDING): Rect[] {
  let acc = [...regions];
  for (const n of nexts) acc = mergeDirty(acc, n, padding);
  return acc;
}
