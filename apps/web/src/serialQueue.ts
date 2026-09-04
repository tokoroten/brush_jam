/**
 * Runs tasks strictly one after another. WebSocket frames arrive in order and
 * must be applied in order; starting an async handler per frame let a slow
 * image load apply an older revision after a newer one. A failing task is
 * reported but never breaks the chain.
 */
export type SerialQueue = (task: () => Promise<void> | void) => void;

export function createSerialQueue(onError: (err: unknown) => void = () => {}): SerialQueue {
  let tail: Promise<void> = Promise.resolve();
  return (task) => {
    tail = tail.then(task).catch(onError);
  };
}
