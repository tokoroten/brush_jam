const ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789';

/**
 * Short client-side stroke id. `crypto.randomUUID` only exists in secure
 * contexts, and Brush Jam is expected to be opened over plain http on a LAN, so
 * this uses `getRandomValues` with a last-resort Math.random fallback.
 */
export function newId(length = 16, random: Crypto | undefined = globalThis.crypto): string {
  const bytes = new Uint8Array(length);
  if (random?.getRandomValues) random.getRandomValues(bytes);
  else for (let i = 0; i < length; i++) bytes[i] = Math.floor(Math.random() * 256);
  let out = '';
  for (let i = 0; i < length; i++) out += ALPHABET[bytes[i]! % ALPHABET.length];
  return out;
}
