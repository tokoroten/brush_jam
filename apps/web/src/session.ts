import { newId } from './id.js';

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const sessionKey = (roomId: string): string => `brushjam.session.${roomId}`;

/**
 * A per-room reconnect token. Stored in sessionStorage so a dropped connection
 * resumes the same server-side identity (and therefore the same undo stack).
 * Storage can be unavailable or throw (private mode, blocked site data), in
 * which case a throwaway token is fine - the user simply gets a new identity.
 */
export function sessionToken(roomId: string, storage: StorageLike | undefined): string {
  const key = sessionKey(roomId);
  try {
    const existing = storage?.getItem(key);
    if (existing) return existing;
    const fresh = newId(24);
    storage?.setItem(key, fresh);
    return fresh;
  } catch {
    return newId(24);
  }
}
