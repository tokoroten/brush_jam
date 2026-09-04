/**
 * Copying the invite URL, which is the one thing that has to work on a LAN.
 *
 * `navigator.clipboard` only exists in a secure context, and the documented
 * playtest URL is plain HTTP on a LAN IP - exactly the case where it is
 * missing, and where a bare `await navigator.clipboard.writeText(...)` throws
 * into a floating promise and the button silently does nothing. So: try the
 * modern API when it is actually there, fall back to a selected off-screen
 * input plus the deprecated `execCommand('copy')`, and let the caller show the
 * URL as text either way so it can always be copied by hand.
 */

export interface CopyDeps {
  clipboard?: { writeText(text: string): Promise<void> } | undefined;
  /** document.execCommand, present in every browser that lacks the above. */
  legacyCopy?: ((text: string) => boolean) | undefined;
}

/** The default legacy path: a selected off-screen input the browser can copy. */
export function domLegacyCopy(doc: Document): (text: string) => boolean {
  return (text: string): boolean => {
    const input = doc.createElement('textarea');
    input.value = text;
    // Off-screen rather than hidden: a display:none element cannot be selected.
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.top = '-1000px';
    input.style.opacity = '0';
    doc.body.appendChild(input);
    try {
      input.select();
      input.setSelectionRange(0, text.length);
      return doc.execCommand('copy');
    } catch {
      return false;
    } finally {
      input.remove();
    }
  };
}

export function browserCopyDeps(): CopyDeps {
  const nav = typeof navigator === 'undefined' ? undefined : navigator;
  // isSecureContext is the actual gate: the API can be *present* and still
  // reject on an insecure origin, so both are checked and the call is guarded.
  const secure = typeof window !== 'undefined' && window.isSecureContext !== false;
  return {
    clipboard: secure && nav?.clipboard ? nav.clipboard : undefined,
    legacyCopy: typeof document === 'undefined' ? undefined : domLegacyCopy(document),
  };
}

/** True if the text reached the clipboard by any route. */
export async function copyText(text: string, deps: CopyDeps): Promise<boolean> {
  if (deps.clipboard) {
    try {
      await deps.clipboard.writeText(text);
      return true;
    } catch {
      /* insecure context, or the user denied permission - try the old way */
    }
  }
  try {
    return deps.legacyCopy?.(text) ?? false;
  } catch {
    return false;
  }
}
