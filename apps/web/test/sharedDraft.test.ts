import { describe, expect, it, vi } from 'vitest';
import {
  DRAFT_DEBOUNCE_MS,
  adoptForeign,
  blurDraft,
  changedDraft,
  editDraft,
  focusDraft,
  initialDraft,
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
    this.sent.push(value);
    this.state = sentDraft(this.state, value, this.server);
    if (value !== this.server) this.inFlight.push(value);
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
