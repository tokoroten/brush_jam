import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * A field whose value is shared by the whole room.
 *
 * The room prompt lives on the server, so what a player types has to travel
 * there and come back. The obvious version of that - "send after 500 ms, then
 * follow the server again" - loses text on a slow link, which is exactly where
 * it matters: sending marks the draft clean, the next render copies the server
 * value (still the *old* one, the echo is 300 ms away) back into the input, and
 * the letters vanish under the cursor. Any other player's change landing in the
 * same window overwrites what is being typed, too.
 *
 * So a draft is only released when the server confirms *this* value:
 *
 * - `edit` marks the draft dirty and keeps it that way through the send.
 * - `sent` remembers what went out. If the server already holds that value it
 *   will never echo, so the draft is released immediately.
 * - `changed` releases the draft when the echo matches what was sent and the
 *   player has not typed on since. Anything else is another player's value:
 *   while the field is being edited it is offered (`foreign`) instead of
 *   applied, and applied directly when it is not.
 */
export interface DraftState<T> {
  /** What the field shows. */
  draft: T;
  /** The player has typed something the server has not confirmed. */
  dirty: boolean;
  /** The last value sent, still waiting for its echo. */
  pending: T | null;
  /** Another player's value, arrived while this field was being edited. */
  foreign: T | null;
  focused: boolean;
}

export const initialDraft = <T,>(value: T): DraftState<T> => ({
  draft: value,
  dirty: false,
  pending: null,
  foreign: null,
  focused: false,
});

/** The player typed. */
export const editDraft = <T,>(state: DraftState<T>, value: T): DraftState<T> => ({
  ...state,
  draft: value,
  dirty: true,
  // What someone else set is not worth offering once this field has moved on.
  foreign: null,
});

/**
 * `value` has just been sent. `server` is what the client believes the server
 * holds: if they are equal there is nothing to echo, so the draft is released
 * now rather than waiting for a message that will never come.
 */
export function sentDraft<T>(state: DraftState<T>, value: T, server: T): DraftState<T> {
  if (Object.is(value, server)) {
    return { ...state, pending: null, dirty: !Object.is(state.draft, value) };
  }
  return { ...state, pending: value };
}

/** The server says the shared value is now `value`. */
export function changedDraft<T>(state: DraftState<T>, value: T): DraftState<T> {
  if (state.pending !== null && Object.is(state.pending, value)) {
    // Our own echo. Still dirty if more was typed while it was in flight.
    return { ...state, pending: null, dirty: !Object.is(state.draft, value), foreign: null };
  }
  const editing = state.focused || state.dirty || state.pending !== null;
  if (editing) return { ...state, foreign: value };
  return { ...state, draft: value, dirty: false, pending: null, foreign: null };
}

/** The player asked to take the other player's value. */
export const adoptForeign = <T,>(state: DraftState<T>): DraftState<T> =>
  state.foreign === null
    ? state
    : { ...state, draft: state.foreign, foreign: null, dirty: false, pending: null };

export const focusDraft = <T,>(state: DraftState<T>): DraftState<T> => ({ ...state, focused: true });
export const blurDraft = <T,>(state: DraftState<T>): DraftState<T> => ({ ...state, focused: false });

export const DRAFT_DEBOUNCE_MS = 500;

export interface SharedDraft<T> {
  value: T;
  dirty: boolean;
  /** Another player's value, waiting to be adopted (null when there is none). */
  foreign: T | null;
  set(value: T): void;
  /** Send now: blur, Enter, or anything else that ends the edit. */
  flush(): void;
  adopt(): void;
  onFocus(): void;
  onBlur(): void;
}

/**
 * The state machine above, wired to a debounce.
 *
 * `server` is the room's current value and `send` puts a new one on the wire;
 * everything else is the caller's markup.
 */
export function useSharedDraft<T>(server: T, send: (value: T) => void): SharedDraft<T> {
  const [state, setState] = useState<DraftState<T>>(() => initialDraft(server));
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Read inside callbacks that must not be re-created on every keystroke.
  const latest = useRef({ state, server, send });
  latest.current = { state, server, send };

  const cancel = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  const flush = useCallback(() => {
    cancel();
    const { state: current, server: known, send: post } = latest.current;
    if (!current.dirty) return;
    if (current.pending !== null && Object.is(current.pending, current.draft)) return;
    post(current.draft);
    setState((s) => sentDraft(s, current.draft, known));
  }, [cancel]);

  const set = useCallback(
    (value: T) => {
      // The ref is updated here as well as through the render, so an Enter in
      // the same tick as the keystroke flushes the letter that was just typed.
      const next = editDraft(latest.current.state, value);
      latest.current.state = next;
      setState(next);
      cancel();
      timer.current = setTimeout(flush, DRAFT_DEBOUNCE_MS);
    },
    [cancel, flush],
  );

  // The room's value changed - our echo, or somebody else's edit.
  useEffect(() => {
    setState((s) => changedDraft(s, server));
  }, [server]);

  useEffect(() => cancel, [cancel]);

  return {
    value: state.draft,
    dirty: state.dirty,
    foreign: state.foreign,
    set,
    flush,
    adopt: () => setState(adoptForeign),
    onFocus: () => setState(focusDraft),
    onBlur: () => {
      flush();
      setState(blurDraft);
    },
  };
}
