import { useEffect, useRef, useState, type JSX } from 'react';
import {
  DEFAULT_EXPORT_FPS,
  EXPORT_FPS_CHOICES,
  EXPORT_VIDEO_HINT,
  EXPORT_ZIP_HINT,
  historyExportAvailability,
  historyVideoUrl,
  historyZipUrl,
} from './exportOptions.js';
import { historyExportFileName } from './gallery.js';

export interface ExportDialogProps {
  roomId: string;
  open: boolean;
  onClose: () => void;
  /** Enough of the gallery state to say whether there is a history to take. */
  history: { enabled: boolean; entries: number; latestN: number | null };
  /** The two pictures this browser makes itself (save.ts). */
  onDownloadDrawing: () => void;
  onDownloadAi: () => void;
}

/**
 * The four ways out of a room, in one place.
 *
 * A native `<dialog>`, so Esc, the focus trap and the inert page behind it are
 * the browser's job rather than four more effects here. The rules of what is
 * offered live in exportOptions.ts; this is the markup and the one piece of
 * state the markup owns, which is the frame rate.
 */
export function ExportDialog({
  roomId,
  open,
  onClose,
  history,
  onDownloadDrawing,
  onDownloadAi,
}: ExportDialogProps): JSX.Element {
  const ref = useRef<HTMLDialogElement>(null);
  const [fps, setFps] = useState<number>(DEFAULT_EXPORT_FPS);
  const { ready, hint } = historyExportAvailability(history);

  useEffect(() => {
    const dialog = ref.current;
    // showModal is missing in jsdom and in the odd embedded browser; a dialog
    // that cannot open modally must still not be a crash on the way to it.
    if (!dialog || typeof dialog.showModal !== 'function') return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  /**
   * A click that lands on the dialog element itself is a click on the backdrop:
   * everything inside is in a child, so the dialog is only ever the target of
   * the padding around it and of the ::backdrop pseudo-element.
   */
  const onBackdrop = (e: React.MouseEvent<HTMLDialogElement>): void => {
    if (e.target === ref.current) onClose();
  };

  return (
    <dialog ref={ref} className="export-dialog" onCancel={onClose} onClose={onClose} onClick={onBackdrop}>
      <div className="export-body">
        <div className="export-head">
          <strong>Export</strong>
          <button onClick={onClose} title="close (Esc)">
            close
          </button>
        </div>

        <div className="export-item">
          {/* Closed afterwards: a failure is reported as a toast, and a modal
              dialog would be sitting on top of it. */}
          <button
            onClick={() => {
              onDownloadDrawing();
              onClose();
            }}
          >
            Download drawing (PNG)
          </button>
          <span className="hint">every visible layer, at canvas size, on white - exactly what is on the stage</span>
        </div>

        <div className="export-item">
          <button
            onClick={() => {
              onDownloadAi();
              onClose();
            }}
          >
            Download AI image (PNG)
          </button>
          <span className="hint">the room&apos;s current AI result, as the server holds it</span>
        </div>

        <div className="export-item">
          {ready ? (
            <a className="export-link" href={historyZipUrl(roomId)} download={historyExportFileName(roomId, 'zip')}>
              Export history as ZIP
            </a>
          ) : (
            <span className="export-link disabled" aria-disabled="true">
              Export history as ZIP
            </span>
          )}
          <span className="hint">{EXPORT_ZIP_HINT}</span>
        </div>

        <div className="export-item">
          <div className="export-row">
            {ready ? (
              <a className="export-link" href={historyVideoUrl(roomId, fps)} download={historyExportFileName(roomId, 'avi')}>
                Download video (Motion JPEG AVI)
              </a>
            ) : (
              <span className="export-link disabled" aria-disabled="true">
                Download video (Motion JPEG AVI)
              </span>
            )}
            <label title="how fast the flip-book plays">
              fps
              <select value={fps} disabled={!ready} onChange={(e) => setFps(Number(e.target.value))}>
                {EXPORT_FPS_CHOICES.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <span className="hint">{EXPORT_VIDEO_HINT}</span>
        </div>

        {hint === null ? null : <p className="hint export-why">{hint}</p>}
      </div>
    </dialog>
  );
}
