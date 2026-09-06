import { useState, type JSX } from 'react';
import type { Layer } from '@brushjam/shared';
import { LAYER_SCALE_STEP, MAX_LAYER_SCALE, MIN_LAYER_SCALE } from './move.js';
import {
  pinOverlay,
  pinnableSource,
  setOverlayOpacity,
  toggleOverlay,
  unpinOverlay,
  type OverlayState,
} from './overlay.js';
import type { RoomClient } from './roomClient.js';

export interface LayerPanelProps {
  client: RoomClient;
  layers: Layer[];
  activeLayerId: string | null;
  maxLayers: number;
  onSelect: (id: string) => void;
  /** Injectable so tests can take both the confirmed and the cancelled path. */
  confirm?: (message: string) => boolean;
  /**
   * The AI overlay (overlay.ts), shown above the layer list.
   *
   * It sits here rather than in the header because that is what it is: a
   * sheet over the drawing, alongside the layers it covers - and the rules of
   * it (what is pinned, what Tab does) belong next to the thing they affect.
   */
  overlay?: {
    state: OverlayState;
    /** The room, for deriving what "pin" would freeze. */
    roomId: string;
    aiRevision: number;
    /** The newest history entry, or null on a server that keeps none. */
    historyN: number | null;
    set: (update: (state: OverlayState) => OverlayState) => void;
  };
}

const browserConfirm = (message: string): boolean =>
  typeof window === 'undefined' ? true : window.confirm(message);

/**
 * Clearing and deleting a layer are destructive and cannot be undone, so both
 * ask first. Exported (and confirm injected) so the guard can be tested without
 * a real dialog.
 */
export function layerActions(
  client: RoomClient,
  confirm: (message: string) => boolean,
): { clear: (layer: Layer) => void; remove: (layer: Layer) => void } {
  return {
    clear: (layer) => {
      if (!confirm(`Clear all strokes on '${layer.name}'? This cannot be undone.`)) return;
      client.send({ t: 'clear_layer', layerId: layer.id });
    },
    remove: (layer) => {
      if (!confirm(`Delete layer '${layer.name}'? This cannot be undone.`)) return;
      client.send({ t: 'layer_delete', id: layer.id });
    },
  };
}

/** Deliberately plain: list is top = frontmost, matching the render order reversed. */
export function LayerPanel({
  client,
  layers,
  activeLayerId,
  maxLayers,
  onSelect,
  confirm = browserConfirm,
  overlay,
}: LayerPanelProps): JSX.Element {
  const [renaming, setRenaming] = useState<string | null>(null);
  const front = [...layers].reverse();

  const reorder = (id: string, delta: number): void => {
    const ids = layers.map((l) => l.id);
    const index = ids.indexOf(id);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= ids.length) return;
    [ids[index], ids[target]] = [ids[target]!, ids[index]!];
    client.send({ t: 'layer_reorder', ids });
  };

  const { clear: clearLayer, remove: deleteLayer } = layerActions(client, confirm);

  // What "pin" would freeze: a followed overlay draws the live AI raster, which
  // has no address, so the button needs one derived for it (overlay.ts).
  const overlayPin = overlay
    ? pinnableSource(overlay.roomId, overlay.aiRevision, overlay.historyN)
    : null;

  return (
    <aside className="layers">
      {overlay ? (
        <div className="overlay-block">
          <div className="layer-row">
            <label title="lay the AI result over the drawing, to trace on">
              <input
                type="checkbox"
                checked={overlay.state.on}
                onChange={() => overlay.set(toggleOverlay)}
              />{' '}
              AI overlay
            </label>
          </div>
          {overlay.state.on ? (
            <>
              <div className="layer-row slider-row">
                <label title="how much of the AI result shows through">
                  <span className="slider-label">opacity</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    aria-label="overlay opacity"
                    value={Math.round(overlay.state.opacity * 100)}
                    onChange={(e) =>
                      overlay.set((o) => setOverlayOpacity(o, Number(e.target.value) / 100))
                    }
                  />
                  <span className="slider-value">{Math.round(overlay.state.opacity * 100)}%</span>
                </label>
              </div>
              <div className="layer-row actions">
                <button
                  className={overlay.state.pinned ? 'active' : ''}
                  disabled={!overlay.state.pinned && overlayPin === null}
                  title={
                    overlay.state.pinned
                      ? 'showing one frozen picture; click to follow the latest result again'
                      : 'freeze the picture on screen, so later generations do not replace it'
                  }
                  onClick={() =>
                    overlay.set((o) => (o.pinned ? unpinOverlay(o) : pinOverlay(o, overlayPin)))
                  }
                >
                  {overlay.state.pinned ? 'pinned' : 'pin'}
                </button>
              </div>
              <p className="hint">Hold Tab to look underneath.</p>
            </>
          ) : null}
        </div>
      ) : null}
      <div className="layers-head">
        <span>Layers</span>
        <button disabled={layers.length >= maxLayers} onClick={() => client.send({ t: 'layer_create', layer: { kind: 'draw' } })}>
          + add
        </button>
      </div>
      <div className="layer-list">
      {front.map((layer) => (
        <div key={layer.id} className={`layer ${layer.id === activeLayerId ? 'active' : ''}`} onClick={() => onSelect(layer.id)}>
          <div className="layer-row">
            <input
              type="checkbox"
              title="visible"
              checked={layer.visible}
              onChange={(e) => client.send({ t: 'layer_update', id: layer.id, patch: { visible: e.target.checked } })}
            />
            {renaming === layer.id ? (
              <input
                autoFocus
                defaultValue={layer.name}
                onBlur={(e) => {
                  client.send({ t: 'layer_update', id: layer.id, patch: { name: e.target.value } });
                  setRenaming(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                }}
              />
            ) : (
              <span className="layer-name" onDoubleClick={() => setRenaming(layer.id)}>
                {layer.name}
                {layer.kind === 'reference' ? ' (ref)' : ''}
              </span>
            )}
            <button title="lock" onClick={() => client.send({ t: 'layer_update', id: layer.id, patch: { locked: !layer.locked } })}>
              {layer.locked ? 'locked' : 'open'}
            </button>
          </div>
          <div className="layer-row slider-row">
            <label title="opacity">
              <span className="slider-label">opacity</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={layer.opacity}
                onChange={(e) => client.send({ t: 'layer_update', id: layer.id, patch: { opacity: Number(e.target.value) } })}
              />
              <span className="slider-value">{layer.opacity.toFixed(2)}</span>
            </label>
          </div>
          <div className="layer-row actions">
            <button onClick={() => reorder(layer.id, 1)}>up</button>
            <button onClick={() => reorder(layer.id, -1)}>down</button>
            <button onClick={() => clearLayer(layer)}>clear</button>
            <button onClick={() => deleteLayer(layer)}>del</button>
          </div>
          {layer.kind === 'reference' && (
            <>
            <div className="layer-row">
              <label title="include this reference in the AI input">
                <input
                  type="checkbox"
                  checked={layer.includeInAI}
                  onChange={(e) => client.send({ t: 'layer_update', id: layer.id, patch: { includeInAI: e.target.checked } })}
                />
                AI input
              </label>
            </div>
            <div className="layer-row slider-row">
              <label title="scale">
                <span className="slider-label">scale</span>
                <input
                  type="range"
                  min={MIN_LAYER_SCALE}
                  max={MAX_LAYER_SCALE}
                  step={LAYER_SCALE_STEP}
                  value={layer.scale ?? 1}
                  onChange={(e) => client.send({ t: 'layer_update', id: layer.id, patch: { scale: Number(e.target.value) } })}
                />
                <span className="slider-value">{(layer.scale ?? 1).toFixed(2)}</span>
              </label>
            </div>
            </>
          )}
        </div>
      ))}
      <p className="hint">
        Space or Shift + drag to pan, wheel to zoom. Ctrl/Cmd+V pastes an image as a reference layer. Undo affects only your own
        strokes.
      </p>
      </div>
    </aside>
  );
}
