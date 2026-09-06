import { describe, expect, it, vi } from 'vitest';
import { MAX_SEED, clampSeed, randomSeed } from '@brushjam/shared';
import {
  DRAFT_DEBOUNCE_MS,
  adoptForeign,
  blurDraft,
  changedDraft,
  editDraft,
  focusDraft,
  initialDraft,
  reconnectedDraft,
  sentDraft,
  type DraftState,
} from '../src/sharedDraft.js';

/**
 * A field on a slow link, driven the way the hook drives it: keystrokes, a
 * 500 ms debounce, and echoes that arrive an RTT later. The bug this replaces
 * cost the user their text on a 300 ms link - sending marked the draft clean,
 * so the next render put the server's *old* value back under the cursor.
 */
class Field {
  state: DraftState<string>;
  server: string;
  readonly sent: string[] = [];
  /** Values the server has accepted but whose echo has not arrived. */
  private readonly inFlight: string[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(server = '') {
    this.server = server;
    this.state = initialDraft(server);
  }

  type(value: string): void {
    this.state = editDraft(this.state, value);
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), DRAFT_DEBOUNCE_MS);
  }

  flush(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    if (!this.state.dirty) return;
    if (this.state.pending !== null && this.state.pending === this.state.draft) return;
    const value = this.state.draft;
    // What the hook does with a send that reports failure: nothing is pending,
    // and the field still owes the room its text. `RoomClient.sendSetting`
    // reports failure for a socket that is up but has no snapshot yet.
    if (!this.online || !this.admitted) return;
    this.sent.push(value);
    this.state = sentDraft(this.state, value, this.server, this.connection);
    // Whatever the field considers pending, the server will echo.
    if (this.state.pending === value) this.inFlight.push(value);
  }

  /** The server processed the oldest send and broadcast prompt_changed. */
  deliverEcho(): void {
    const value = this.inFlight.shift();
    if (value === undefined) return;
    this.server = value;
    this.state = changedDraft(this.state, value);
  }

  /** Another player set the value. */
  foreignChange(value: string): void {
    this.server = value;
    this.state = changedDraft(this.state, value);
  }

  /** The socket is down: `RoomClient.send` reports false and nothing goes. */
  online = true;

  /** Which socket is on the wire (`RoomClient.connectionEpoch`). */
  connection = 1;
  /** Whether this socket's snapshot has arrived (`RoomClient.admitted`). */
  admitted = true;

  /** The socket came back. Its snapshot has not arrived, so nothing is sent. */
  newSocket(): void {
    this.online = true;
    this.admitted = false;
    this.connection += 1;
    this.inFlight.length = 0; // nothing from the dead socket will be echoed
  }

  /**
   * A snapshot arrived on the current socket, carrying whatever the room
   * actually holds.
   *
   * The order is the hook's: the server-value effect reconciles with the new
   * snapshot first, then the epoch effect decides what is still owed, then the
   * flush pays it.
   */
  snapshot(server: string): void {
    this.admitted = true;
    const changed = server !== this.server;
    this.server = server;
    if (changed) this.state = changedDraft(this.state, server);
    this.state = reconnectedDraft(this.state, server, this.connection);
    this.flush();
  }

  /** The usual case: a new socket, then its snapshot. */
  reconnect(server: string): void {
    this.newSocket();
    this.snapshot(server);
  }
}

/** The same field, holding a number - the seed control. */
class NumberField {
  state: DraftState<number>;
  server: number;
  readonly sent: number[] = [];
  private readonly inFlight: number[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(server: number) {
    this.server = server;
    this.state = initialDraft(server);
  }

  type(value: number): void {
    this.state = editDraft(this.state, value);
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), DRAFT_DEBOUNCE_MS);
  }

  flush(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    if (!this.state.dirty) return;
    if (this.state.pending !== null && this.state.pending === this.state.draft) return;
    const value = this.state.draft;
    this.sent.push(value);
    this.state = sentDraft(this.state, value, this.server);
    if (value !== this.server) this.inFlight.push(value);
  }

  /** The dice button: set and send, without waiting out the debounce. */
  roll(value: number): void {
    this.type(value);
    this.flush();
  }

  deliverEcho(): void {
    const value = this.inFlight.shift();
    if (value === undefined) return;
    this.server = value;
    this.state = changedDraft(this.state, value);
  }

  foreignChange(value: number): void {
    this.server = value;
    this.state = changedDraft(this.state, value);
  }
}

describe('a room-wide field being typed into', () => {
  it('never shows the old server value between the send and its echo', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('');
      field.type('anime');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['anime']);
      // 300 ms of round trip, during which the server still says ''.
      expect(field.state.draft).toBe('anime');
      expect(field.state.dirty).toBe(true);
      field.foreignChange(''); // a re-broadcast of the value we already have
      expect(field.state.draft).toBe('anime');

      field.server = '';
      field.deliverEcho();
      expect(field.state.draft).toBe('anime');
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBe(null);
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps letters typed while the previous value is still in flight', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('');
      field.type('anime');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.type('anime style'); // still typing; "anime" has not come back yet
      field.deliverEcho(); // the echo of "anime"

      expect(field.state.draft).toBe('anime style');
      expect(field.state.dirty).toBe(true);
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['anime', 'anime style']);

      field.deliverEcho();
      expect(field.state.draft).toBe('anime style');
      expect(field.state.dirty).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('offers another player’s value instead of overwriting the draft', () => {
    const field = new Field('');
    field.state = focusDraft(field.state);
    field.type('anime');
    field.foreignChange('fantasy town');

    expect(field.state.draft).toBe('anime');
    expect(field.state.foreign).toBe('fantasy town');

    field.state = adoptForeign(field.state);
    expect(field.state.draft).toBe('fantasy town');
    expect(field.state.foreign).toBe(null);
    expect(field.state.dirty).toBe(false);
  });

  it('applies another player’s value directly when the field is idle', () => {
    const field = new Field('');
    field.foreignChange('fantasy town');
    expect(field.state.draft).toBe('fantasy town');
    expect(field.state.foreign).toBe(null);
    expect(field.state.dirty).toBe(false);
  });

  it('sends immediately on Enter or blur instead of waiting out the debounce', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('');
      field.type('anime');
      field.flush(); // Enter
      expect(field.sent).toEqual(['anime']);

      // The cancelled debounce must not fire a second send.
      vi.advanceTimersByTime(2 * DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['anime']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('releases the draft at once when the value it sent is what the server holds', () => {
    // No echo is coming: the server has nothing to change, so waiting for one
    // would leave the field dirty forever.
    const field = new Field('anime');
    field.state = editDraft(field.state, 'anim');
    field.state = editDraft(field.state, 'anime');
    field.flush();
    expect(field.state.dirty).toBe(false);
    expect(field.state.pending).toBe(null);
  });

  it('stays dirty when the server echoes a value that is not the one we sent', () => {
    const field = new Field('');
    field.type('anime');
    field.flush();
    field.foreignChange('someone else'); // not our echo
    expect(field.state.draft).toBe('anime');
    expect(field.state.pending).toBe('anime');
    expect(field.state.foreign).toBe('someone else');
  });

  it('keeps the offer standing while the player carries on typing', () => {
    // It used to vanish on the next keystroke, which took away both the notice
    // and any way of finding out what the other player had set.
    const field = new Field('');
    field.type('anime');
    field.foreignChange('fantasy town');
    field.type('anime style');
    expect(field.state.foreign).toBe('fantasy town');
    expect(field.state.draft).toBe('anime style');
  });

  it('drops the offer once the room shows our value instead', () => {
    const field = new Field('');
    field.state = focusDraft(field.state);
    field.type('anime');
    field.foreignChange('fantasy town');
    field.flush();
    field.deliverEcho(); // the room is ours now; their value is history
    expect(field.state.foreign).toBe(null);
    expect(field.state.draft).toBe('anime');
    expect(field.state.dirty).toBe(false);
  });

  it('drops the offer when the room arrives at what we already show', () => {
    const field = new Field('');
    field.type('anime');
    field.foreignChange('fantasy town');
    field.foreignChange('anime'); // someone typed the same thing
    expect(field.state.foreign).toBe(null);
    expect(field.state.draft).toBe('anime');
    expect(field.state.dirty).toBe(false);
  });

  it('takes the other player’s value on blur when nothing is being edited', () => {
    // Enter with the cursor still in the field: the change is sent, and a
    // later change by someone else can only be offered. Leaving on blur is
    // what says the field is a view of the room again.
    const field = new Field('');
    field.state = focusDraft(field.state);
    field.type('anime');
    field.flush();
    field.deliverEcho();
    field.foreignChange('fantasy town');
    expect(field.state.draft).toBe('anime');
    expect(field.state.foreign).toBe('fantasy town');

    field.state = blurDraft(field.state);
    expect(field.state.draft).toBe('fantasy town');
    expect(field.state.foreign).toBe(null);
  });

  it('keeps an unsent draft through a blur', () => {
    const field = new Field('');
    field.state = focusDraft(field.state);
    field.type('anime');
    field.foreignChange('fantasy town');
    field.state = blurDraft(field.state); // dirty: the draft is still the player's
    expect(field.state.draft).toBe('anime');
    expect(field.state.foreign).toBe('fantasy town');
  });

  it('lets an unfocused, unedited field follow the room again after a blur', () => {
    const field = new Field('');
    field.state = focusDraft(field.state);
    field.type('anime');
    field.flush();
    field.deliverEcho();
    field.state = blurDraft(field.state);
    field.foreignChange('fantasy town');
    expect(field.state.draft).toBe('fantasy town');
  });

  it('works the same for a number, which is what the denoise slider needs', () => {
    let state = initialDraft(0.55);
    state = editDraft(state, 0.8);
    state = sentDraft(state, 0.8, 0.55);
    expect(state.pending).toBe(0.8);
    state = changedDraft(state, 0.8);
    expect(state).toMatchObject({ draft: 0.8, dirty: false, pending: null });
  });
});

/**
 * The seed field is the same machine with numbers in it, plus a dice: the
 * whole point of a fixed seed is being able to ask for a different picture
 * from the same drawing, and that must not wait out a debounce.
 */
describe('the seed control', () => {
  it('sends a re-roll immediately, with no debounce', () => {
    vi.useFakeTimers();
    try {
      const field = new NumberField(1234);
      field.roll(555); // what the dice button does: set, then flush
      expect(field.sent).toEqual([555]);
      expect(field.state.draft).toBe(555);
      vi.advanceTimersByTime(2 * DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual([555]); // and not twice
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps a typed seed until it is flushed', () => {
    vi.useFakeTimers();
    try {
      const field = new NumberField(1234);
      field.type(42);
      expect(field.sent).toEqual([]);
      field.flush(); // Enter or blur
      expect(field.sent).toEqual([42]);
      field.deliverEcho();
      expect(field.state.dirty).toBe(false);
      expect(field.state.draft).toBe(42);
    } finally {
      vi.useRealTimers();
    }
  });

  it('offers another player’s re-roll instead of stealing the one being typed', () => {
    const field = new NumberField(1234);
    field.state = focusDraft(field.state);
    field.type(42);
    field.foreignChange(999);
    expect(field.state.draft).toBe(42);
    expect(field.state.foreign).toBe(999);

    field.state = adoptForeign(field.state);
    expect(field.state.draft).toBe(999);
  });

  it('clamps whatever is typed into the range', () => {
    expect(clampSeed(-1)).toBe(MAX_SEED);
    expect(clampSeed(MAX_SEED + 1)).toBe(0);
    expect(clampSeed(12.7)).toBe(12);
    expect(clampSeed(0)).toBe(0);
    expect(clampSeed(MAX_SEED)).toBe(MAX_SEED);
  });

  it('rolls inside the range', () => {
    const rolls = Array.from({ length: 200 }, () => randomSeed());
    expect(rolls.every((s) => Number.isInteger(s) && s >= 0 && s <= MAX_SEED)).toBe(true);
    expect(new Set(rolls).size).toBeGreaterThan(1);
  });
});

/**
 * Review 2 finding 2: the room reaching the draft's value is only settlement
 * if nothing of ours is still in flight. An older write of our own lands after
 * it, so the obligation to send the newest text stands.
 */
describe('a foreign value equal to the draft, with a write still in flight', () => {
  it('does not release the draft, and the newest text still reaches the room', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('x');
      field.type('A');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['A']); // A is on the wire, unacknowledged
      field.type('B'); // typed before A's echo comes back

      field.foreignChange('B'); // another player happens to set B
      expect(field.state.dirty).toBe(true);
      expect(field.state.pending).toBe('A');
      expect(field.state.foreign).toBeNull(); // nothing to offer: it IS the draft

      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['A', 'B']);

      // The server applied B (theirs), then A (ours), then B (ours again), and
      // broadcasts in that order. The field ends where the last person to type
      // left it, which is what was lost before: with no send of B, the room
      // itself would have stopped at A.
      field.foreignChange('A');
      field.foreignChange('B');
      expect(field.state.draft).toBe('B');
      expect(blurDraft(field.state).draft).toBe('B');
    } finally {
      vi.useRealTimers();
    }
  });

  it('still settles when the room reaches the draft with nothing in flight', () => {
    const settled = changedDraft(editDraft(initialDraft('x'), 'B'), 'B');
    expect(settled.dirty).toBe(false);
    expect(settled.pending).toBeNull();
    expect(settled.foreign).toBeNull();
  });
});

/**
 * Review 2 finding 3: RoomClient.send used to drop a message when the socket
 * was not open, while flush recorded it as pending. Typing during a
 * disconnect left a draft that was waiting for an echo nobody would ever
 * send, and blur returned early because pending equalled the draft.
 */
describe('editing while the connection is down', () => {
  it('keeps the text owed and sends it after the reconnect', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.online = false;
      field.type('a hill with a house');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual([]); // nothing left the machine
      expect(field.state.pending).toBeNull(); // and nothing is pretending it did
      expect(field.state.dirty).toBe(true);

      // The snapshot still says what the room had before the drop.
      field.reconnect('a hill');
      expect(field.sent).toEqual(['a hill with a house']);
      expect(field.state.draft).toBe('a hill with a house');
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not resend what the room already got before the socket died', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.type('a hill with a house');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['a hill with a house']);
      // The send landed; the echo did not, because the socket died first. The
      // snapshot proves it landed.
      field.reconnect('a hill with a house');
      expect(field.sent).toEqual(['a hill with a house']);
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('forgets a pending value from a connection that is gone', () => {
    const state = sentDraft(editDraft(initialDraft('x'), 'B'), 'B', 'x', 1);
    expect(state.pending).toBe('B');
    const after = reconnectedDraft(state, 'x', 2); // a different socket
    expect(after.pending).toBeNull();
    expect(after.dirty).toBe(true); // the room does not have it: send it again
  });
});

/**
 * Review 3 finding 1: the reconnect effect used to mark the field dirty
 * whenever the draft differed from the snapshot, and send. A client mounts
 * with its own defaults, so the first snapshot always differed - and joining a
 * room wrote the joiner's empty prompt over the room's.
 */
describe('joining and reconnecting without an edit', () => {
  it('never sends anything on a fresh join', () => {
    vi.useFakeTimers();
    try {
      // Mounted before the snapshot: the client's own defaults.
      const field = new Field('');
      field.reconnect('a hill in the rain');
      expect(field.sent).toEqual([]);
      expect(field.state.draft).toBe('a hill in the rain');
      expect(field.state.dirty).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('adopts the room silently when the socket dropped with nothing typed', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      // Somebody else changed the prompt while this client was away.
      field.reconnect('a castle');
      expect(field.sent).toEqual([]);
      expect(field.state.draft).toBe('a castle');
      expect(field.state.dirty).toBe(false);
      expect(field.state.foreign).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('sends exactly the one edit that never made it out', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.online = false;
      field.type('a hill with a house');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      // Meanwhile the room moved on without us.
      field.reconnect('a castle');
      expect(field.sent).toEqual(['a hill with a house']);
      expect(field.state.draft).toBe('a hill with a house');
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not rewrite a focused field from under the cursor', () => {
    const state = focusDraft(initialDraft('a hill'));
    const after = reconnectedDraft(state, 'a castle');
    expect(after.draft).toBe('a hill');
    expect(after.foreign).toBe('a castle');
    expect(after.dirty).toBe(false); // and therefore nothing is sent
  });

  it('leaves a settled field alone', () => {
    const after = reconnectedDraft(initialDraft('a hill'), 'a hill');
    expect(after.dirty).toBe(false);
    expect(after.pending).toBeNull();
  });
});


/**
 * Review 4 finding 1: a socket is open, and `RoomClient.send` succeeds, for a
 * moment before its snapshot arrives. Reconciliation assumed every write in
 * flight had died with the previous socket, so a write made on the *new* one
 * was dropped - and its echo then arrived as "somebody else's value" and
 * overwrote what had been typed since.
 */
describe('a write made on the new socket before its snapshot', () => {
  it('waits for the snapshot and then sends only the latest text', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.online = false;
      field.newSocket(); // the socket is up; the snapshot is not here yet
      field.type('a hill at dawn');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual([]); // nothing is written against an unread room
      expect(field.state.pending).toBeNull();
      field.type('a hill at dusk'); // typed on, still before the snapshot

      field.snapshot('a hill at dusk');
      // The room already holds what the field shows, so there is nothing owed.
      expect(field.sent).toEqual([]);
      expect(field.state.draft).toBe('a hill at dusk');
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('pays a pre-snapshot edit exactly once, after the snapshot', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.newSocket();
      field.type('a castle');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual([]);
      field.snapshot('a hill'); // the room still holds the old value
      expect(field.sent).toEqual(['a castle']);
      // Nothing resends it: the debounce, a blur, another render.
      field.flush();
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS * 4);
      expect(field.sent).toEqual(['a castle']);
      field.deliverEcho();
      expect(field.state.draft).toBe('a castle');
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('still drops a write the previous socket swallowed', () => {
    const sent = sentDraft(editDraft(initialDraft('x'), 'B'), 'B', 'x', 3);
    const after = reconnectedDraft(sent, 'x', 4);
    expect(after.pending).toBeNull();
    expect(after.dirty).toBe(true);
  });
});


/**
 * Review 5 finding 3: a write the server has no reason to echo left the field
 * pending for ever - and a field that is for ever dirty never adopts anybody
 * else's value again, so the prompt input stopped following the room.
 */
describe('a write the server will not echo', () => {
  it('does not leave the field waiting for an echo that cannot come', () => {
    vi.useFakeTimers();
    try {
      // The room holds B. A is typed and sent, and the socket dies with it.
      const field = new Field('b');
      field.type('a');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.state.pending).toBe('a');
      field.newSocket(); // a new socket, whose snapshot has not arrived

      // The player puts B back before the snapshot lands.
      field.type('b');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.snapshot('b');

      expect(field.state.pending).toBeNull();
      expect(field.state.dirty).toBe(false);

      // ...and the field follows the room again.
      field.foreignChange('c');
      expect(field.state.draft).toBe('c');
    } finally {
      vi.useRealTimers();
    }
  });

  it('settles a write the snapshot already carries, even mid-edit', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.newSocket();
      field.type('a castle');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      // The snapshot was taken after the server accepted it: no echo is due,
      // because the value never changed again.
      field.snapshot('a castle');
      expect(field.state.pending).toBeNull();
      expect(field.state.dirty).toBe(false);
      field.foreignChange('a castle in the rain');
      expect(field.state.draft).toBe('a castle in the rain');
    } finally {
      vi.useRealTimers();
    }
  });

  it('takes the room value on blur rather than staying stuck', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('b');
      field.state = focusDraft(field.state);
      field.type('a');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.newSocket();
      field.type('b');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.snapshot('b');
      // Somebody else sets something while this field is focused: offered.
      field.foreignChange('c');
      expect(field.state.foreign).toBe('c');
      field.state = blurDraft(field.state);
      expect(field.state.draft).toBe('c');
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not count a dead connection write when deciding an echo is due', () => {
    // Sent on connection 1, lost; the same value sent again on connection 2
    // when the server already holds it is settled at once.
    const lost = sentDraft(editDraft(initialDraft('b'), 'a'), 'a', 'b', 1);
    expect(lost.pending).toBe('a');
    const typedBack = editDraft(lost, 'b'); // the player puts the room's value back
    const resent = sentDraft(typedBack, 'b', 'b', 2);
    expect(resent.pending).toBeNull();
    expect(resent.dirty).toBe(false);
  });
});


/**
 * Review 6: the structural fix. A socket carries messages from `onopen`, but
 * this client does not know what the room holds until its snapshot arrives, so
 * a shared setting written in that window is a write against a value nobody
 * has read. Every reconnect bug in reviews 4 and 5 was a version of that, so
 * the fields do not write until they have been admitted.
 */
describe('nothing is written against a room that has not been read', () => {
  it('ends on the last value typed, whatever went out before the snapshot', () => {
    vi.useFakeTimers();
    try {
      // The reported sequence: the room holds B; on the new socket, before its
      // snapshot, A is submitted and then B.
      const field = new Field('b');
      field.newSocket();
      field.type('a');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.type('b');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual([]); // neither of them left

      field.snapshot('b'); // ...and the snapshot changes nothing
      expect(field.state.draft).toBe('b');
      expect(field.state.foreign).toBeNull(); // no offer of a value we abandoned
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBeNull();

      // Nothing arrives later to undo it, because nothing was ever sent.
      expect(field.sent).toEqual([]);
      expect(field.state.draft).toBe('b');
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not resend a dead connection write when the room already has it', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.type('a castle');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      expect(field.sent).toEqual(['a castle']); // it went out, then the socket died
      field.newSocket();
      // The snapshot proves it landed: nothing is owed and nothing is resent.
      field.snapshot('a castle');
      expect(field.sent).toEqual(['a castle']);
      expect(field.state.dirty).toBe(false);
      expect(field.state.pending).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('resends a dead connection write only while the draft still owes it', () => {
    vi.useFakeTimers();
    try {
      const field = new Field('a hill');
      field.type('a castle');
      vi.advanceTimersByTime(DRAFT_DEBOUNCE_MS);
      field.newSocket();
      field.snapshot('a hill'); // the write never landed
      expect(field.sent).toEqual(['a castle', 'a castle']);
      expect(field.state.pending).toBe('a castle');
    } finally {
      vi.useRealTimers();
    }
  });
});
