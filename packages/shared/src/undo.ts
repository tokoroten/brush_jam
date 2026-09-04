export interface UndoableStroke { id: string; userId: string }

/**
 * Collaborative undo: undo the *sender's* latest not-yet-undone stroke, not the
 * latest global one. (Alice #100, Bob #101, Alice #102 -> Alice's undo hits #102.)
 * Returns null when the user has nothing left to undo.
 */
export function selectUndoTarget<T extends UndoableStroke>(
  strokes: readonly T[],
  undone: ReadonlySet<string>,
  userId: string,
): T | null {
  for (let i = strokes.length - 1; i >= 0; i--) {
    const s = strokes[i]!;
    if (s.userId === userId && !undone.has(s.id)) return s;
  }
  return null;
}
