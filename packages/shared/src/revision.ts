/**
 * Latest-useful-wins: a generation result rendered from an older human revision
 * than the one already composited into the AI canvas is dropped.
 */
export function shouldAcceptResult(forRevision: number, lastAcceptedAIRevision: number): boolean {
  return forRevision >= lastAcceptedAIRevision;
}
