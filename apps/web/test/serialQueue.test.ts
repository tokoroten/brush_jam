import { describe, expect, it } from 'vitest';
import { createSerialQueue } from '../src/serialQueue.js';
import { sessionKey, sessionToken, type StorageLike } from '../src/session.js';

const tick = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Finding 8: ordered frames must be applied in order, even with slow handlers. */
describe('createSerialQueue', () => {
  it('runs tasks in submission order despite differing durations', async () => {
    const done: number[] = [];
    const enqueue = createSerialQueue();
    enqueue(async () => {
      await tick(30);
      done.push(1);
    });
    enqueue(async () => {
      await tick(1);
      done.push(2);
    });
    enqueue(() => {
      done.push(3);
    });
    await tick(80);
    expect(done).toEqual([1, 2, 3]);
  });

  it('keeps processing after a task throws, and reports it', async () => {
    const done: number[] = [];
    const errors: unknown[] = [];
    const enqueue = createSerialQueue((err) => errors.push(err));
    enqueue(() => {
      throw new Error('boom');
    });
    enqueue(async () => {
      await tick(1);
      done.push(2);
    });
    enqueue(() => {
      done.push(3);
    });
    await tick(50);
    expect(done).toEqual([2, 3]);
    expect((errors[0] as Error).message).toBe('boom');
  });

  it('keeps processing after a rejected promise', async () => {
    const done: number[] = [];
    const enqueue = createSerialQueue();
    enqueue(async () => {
      await Promise.reject(new Error('nope'));
    });
    enqueue(() => {
      done.push(1);
    });
    await tick(20);
    expect(done).toEqual([1]);
  });

  it('never overlaps two tasks', async () => {
    let running = 0;
    let maxConcurrent = 0;
    const enqueue = createSerialQueue();
    for (let i = 0; i < 5; i++) {
      enqueue(async () => {
        running += 1;
        maxConcurrent = Math.max(maxConcurrent, running);
        await tick(5);
        running -= 1;
      });
    }
    await tick(120);
    expect(maxConcurrent).toBe(1);
  });
});

/** Finding 11: the reconnect token is stable per room and per session. */
describe('sessionToken', () => {
  const fakeStorage = (): StorageLike & { data: Map<string, string> } => {
    const data = new Map<string, string>();
    return {
      data,
      getItem: (k) => data.get(k) ?? null,
      setItem: (k, v) => void data.set(k, v),
    };
  };

  it('returns the same token for the same room', () => {
    const storage = fakeStorage();
    const first = sessionToken('roomone', storage);
    expect(sessionToken('roomone', storage)).toBe(first);
    expect(storage.data.get(sessionKey('roomone'))).toBe(first);
  });

  it('uses a different token per room', () => {
    const storage = fakeStorage();
    expect(sessionToken('roomone', storage)).not.toBe(sessionToken('roomtwo', storage));
  });

  it('produces a token the server will accept', () => {
    expect(sessionToken('roomone', fakeStorage())).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
  });

  it('still returns a token when storage is unavailable or throws', () => {
    expect(sessionToken('roomone', undefined)).toHaveLength(24);
    const hostile: StorageLike = {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    };
    expect(sessionToken('roomone', hostile)).toHaveLength(24);
  });
});
