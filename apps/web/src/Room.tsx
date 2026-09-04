import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type JSX } from 'react';
import {
  AI_PROFILES,
  AI_RESOLUTIONS,
  CANVAS_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  DENOISE_STEP,
  MAX_DENOISE,
  MAX_LAYERS,
  MAX_NEGATIVE_PROMPT,
  MIN_DENOISE,
  PROFILE_HINT_MS,
  fitCamera,
  panBy,
  screenToWorld,
  zoomAt,
  type AIProfileName,
  type Camera,
  type Layer,
  type Point,
} from '@brushjam/shared';
import { LayerPanel } from './LayerPanel.js';
import { layerOrigin, layerPoint, movePatch, movedPosition, pickMovableLayer, scaledBy } from './move.js';
import { newId } from './id.js';
import { StageView } from './StageView.js';
import { ACCEPTED_PASTE_TYPES, downscaleBlob, pasteLimit, pastePlacement } from './paste.js';
import { RoomClient } from './roomClient.js';

export type Tool = 'pen' | 'noise' | 'eraser' | 'move';

const CURSOR_INTERVAL_MS = 50;
const CHUNK_INTERVAL_MS = 40;
/** ~20 Hz while dragging a reference layer. */
const MOVE_INTERVAL_MS = 50;

interface Drag {
  kind: 'stroke' | 'pan' | 'move';
  /** Raw id sent to the server. */
  strokeId?: string;
  /** Server-qualified id, which is how the live map and every relay key it. */
  liveKey?: string;
  layerId?: string;
  lastScreen: { x: number; y: number };
  sentPoints: number;
  lastChunkAt: number;
  origin?: { x: number; y: number };
}

export function Room({ roomId, name }: { roomId: string; name: string }): JSX.Element {
  const client = useMemo(() => new RoomClient(roomId, name), [roomId, name]);
  const version = useSyncExternalStore(client.subscribe, client.getVersion);

  useEffect(() => {
    client.connect();
    return () => client.dispose();
  }, [client]);

  const [camera, setCamera] = useState<Camera>({ centerX: CANVAS_SIZE / 2, centerY: CANVAS_SIZE / 2, zoom: 0.25 });
  const [tool, setTool] = useState<Tool>('pen');
  const [color, setColor] = useState('#1b1b1b');
  const [width, setWidth] = useState(14);
  const [activeLayerId, setActiveLayerId] = useState<string | null>(null);
  const [promptDraft, setPromptDraft] = useState('');
  const [promptDirty, setPromptDirty] = useState(false);
  const [copied, setCopied] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [denoiseDraft, setDenoiseDraft] = useState<number | null>(null);
  const [negativeDraft, setNegativeDraft] = useState<string | null>(null);

  const dragRef = useRef<Drag | null>(null);
  /** Image id of a paste we are still waiting for the server to turn into a layer. */
  const pastedImageId = useRef<string | null>(null);
  /** Full-precision wheel scaling, throttled to the same rate as a move. */
  const scaleRef = useRef<{ id: string; scale: number; sentAt: number } | null>(null);
  const scaleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const spaceRef = useRef(false);
  const lastCursorAt = useRef(0);

  const layers = client.orderedLayers;
  const sizeOf = (imageId: string): { width: number; height: number } | undefined => {
    const img = client.images.get(imageId);
    return img ? { width: img.naturalWidth, height: img.naturalHeight } : undefined;
  };
  const activeLayer: Layer | undefined =
    layers.find((l) => l.id === activeLayerId) ?? layers.filter((l) => l.kind === 'draw').at(-1) ?? layers.at(-1);

  useEffect(() => {
    if (!promptDirty) setPromptDraft(client.prompt);
  }, [client.prompt, promptDirty]);

  useEffect(() => {
    if (!promptDirty) return;
    const timer = setTimeout(() => {
      client.send({ t: 'set_prompt', prompt: promptDraft });
      setPromptDirty(false);
    }, 500);
    return () => clearTimeout(timer);
  }, [promptDraft, promptDirty, client]);

  // Advanced settings are room-level, so they debounce and echo back exactly
  // like the prompt: a local draft wins until the server confirms it.
  useEffect(() => {
    if (denoiseDraft === null) return;
    const timer = setTimeout(() => {
      client.send({ t: 'set_ai_settings', denoise: denoiseDraft });
      setDenoiseDraft(null);
    }, 500);
    return () => clearTimeout(timer);
  }, [denoiseDraft, client]);

  useEffect(() => {
    if (negativeDraft === null) return;
    const timer = setTimeout(() => {
      client.send({ t: 'set_ai_settings', negativePrompt: negativeDraft });
      setNegativeDraft(null);
    }, 500);
    return () => clearTimeout(timer);
  }, [negativeDraft, client]);

  // The server assigns the layer id, so selection has to wait for the echo.
  useEffect(() => {
    if (!pastedImageId.current) return;
    const created = client.layers.find((l) => l.imageId === pastedImageId.current);
    if (!created) return;
    pastedImageId.current = null;
    setActiveLayerId(created.id);
  }, [client.layers, version]);

  // --- camera helpers -------------------------------------------------------
  const stageSize = (e: { currentTarget: HTMLCanvasElement }): { w: number; h: number } => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { w: rect.width, h: rect.height };
  };
  const toWorld = (e: React.PointerEvent<HTMLCanvasElement> | React.WheelEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    const { w, h } = { w: rect.width, h: rect.height };
    return screenToWorld(camera, w, h, e.clientX - rect.left, e.clientY - rect.top);
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>): void => {
    // Move tool: the wheel scales the layer being moved instead of the camera.
    const moving = dragRef.current?.kind === 'move' ? dragRef.current.layerId : null;
    if (tool === 'move' && (moving || e.shiftKey)) {
      const target = client.findLayer(moving ?? activeLayer?.id ?? '');
      if (target?.kind === 'reference' && !target.locked) {
        // Accumulated locally at full precision, sent at the same ~20 Hz as a
        // move, so a fast wheel does not become a burst of layer_updates.
        const base = scaleRef.current?.id === target.id ? scaleRef.current.scale : (target.scale ?? 1);
        const scale = scaledBy(base, Math.exp(-e.deltaY * 0.0015));
        const now = Date.now();
        const last = scaleRef.current?.id === target.id ? scaleRef.current.sentAt : 0;
        scaleRef.current = { id: target.id, scale, sentAt: now - last > MOVE_INTERVAL_MS ? now : last };
        if (now - last > MOVE_INTERVAL_MS) client.send({ t: 'layer_update', id: target.id, patch: { scale } });
        else {
          if (scaleTimer.current) clearTimeout(scaleTimer.current);
          scaleTimer.current = setTimeout(() => {
            const pending = scaleRef.current;
            if (pending) client.send({ t: 'layer_update', id: pending.id, patch: { scale: pending.scale } });
          }, MOVE_INTERVAL_MS);
        }
        return;
      }
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const { w, h } = stageSize(e);
    setCamera((cam) => zoomAt(cam, w, h, e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015)));
  };

  const fit = (): void => {
    const el = document.querySelector('.stage');
    const rect = el?.getBoundingClientRect();
    setCamera(fitCamera(rect?.width ?? 800, rect?.height ?? 600, client.canvasSize));
  };
  const reset = (): void => setCamera((cam) => ({ ...cam, zoom: 1 }));

  // The world size comes from the snapshot, so the default view can only be
  // fitted once the server has told us how big the canvas is.
  const fittedFor = useRef(0);
  useEffect(() => {
    if (fittedFor.current === client.canvasSize) return;
    fittedFor.current = client.canvasSize;
    const rect = document.querySelector('.stage')?.getBoundingClientRect();
    setCamera(fitCamera(rect?.width ?? 800, rect?.height ?? 600, client.canvasSize));
  }, [client.canvasSize]);

  // --- drawing --------------------------------------------------------------
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const screen = { x: e.clientX, y: e.clientY };
    const world = toWorld(e);

    if (e.button === 1 || spaceRef.current || e.shiftKey) {
      dragRef.current = { kind: 'pan', lastScreen: screen, sentPoints: 0, lastChunkAt: 0 };
      return;
    }
    if (tool === 'move') {
      // Topmost reference under the pointer wins; otherwise the selected one,
      // which is what makes a freshly pasted image draggable straight away.
      const target = pickMovableLayer(
        layers,
        { sizeOf, strokes: client.strokes, undone: client.undone },
        world,
        activeLayer?.id ?? null,
      );
      if (target) {
        setActiveLayerId(target.id);
        dragRef.current = {
          kind: 'move',
          layerId: target.id,
          lastScreen: screen,
          sentPoints: 0,
          lastChunkAt: 0,
          origin: layerOrigin(target),
        };
      }
      // the move tool never draws, even with nothing to move
      return;
    }
    if (!activeLayer || activeLayer.kind !== 'draw' || activeLayer.locked) return;

    const strokeId = newId();
    // The server namespaces ids by author; mirror that locally so the committed
    // stroke replaces the live one instead of leaving a duplicate behind.
    const liveKey = `${client.youUserId}:${strokeId}`;
    // The layer is rendered translated, so points are recorded in layer space
    // and the line appears exactly under the pointer.
    const point: Point = { ...layerPoint(world, activeLayer), p: e.pressure > 0 ? e.pressure : 1 };
    const init = { id: liveKey, layerId: activeLayer.id, tool: tool === 'eraser' ? ('eraser' as const) : tool === 'noise' ? ('noise' as const) : ('pen' as const), color, width, points: [point] };
    client.live.set(liveKey, { userId: client.youUserId, init, points: [point] });
    client.send({ t: 'stroke_start', stroke: { ...init, id: strokeId } });
    dragRef.current = { kind: 'stroke', strokeId, liveKey, lastScreen: screen, sentPoints: 1, lastChunkAt: Date.now() };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>): void => {
    const world = toWorld(e);
    const now = Date.now();
    if (now - lastCursorAt.current > CURSOR_INTERVAL_MS) {
      lastCursorAt.current = now;
      client.send({ t: 'cursor', x: world.x, y: world.y });
    }

    const drag = dragRef.current;
    if (!drag) return;

    if (drag.kind === 'pan') {
      const dx = e.clientX - drag.lastScreen.x;
      const dy = e.clientY - drag.lastScreen.y;
      drag.lastScreen = { x: e.clientX, y: e.clientY };
      setCamera((cam) => panBy(cam, dx, dy));
      return;
    }
    if (drag.kind === 'move' && drag.layerId && drag.origin) {
      drag.origin = movedPosition(drag.origin, e.clientX - drag.lastScreen.x, e.clientY - drag.lastScreen.y, camera.zoom);
      drag.lastScreen = { x: e.clientX, y: e.clientY };
      if (now - drag.lastChunkAt > MOVE_INTERVAL_MS) {
        drag.lastChunkAt = now;
        const moved = client.findLayer(drag.layerId);
        if (moved) client.send({ t: 'layer_update', id: drag.layerId, patch: movePatch(moved, drag.origin) });
      }
      return;
    }
    if (drag.kind === 'stroke' && drag.strokeId && drag.liveKey) {
      const live = client.live.get(drag.liveKey);
      if (!live) return;
      const owner = client.findLayer(live.init.layerId);
      live.points.push({ ...layerPoint(world, owner), p: e.pressure > 0 ? e.pressure : 1 });
      if (now - drag.lastChunkAt > CHUNK_INTERVAL_MS) {
        drag.lastChunkAt = now;
        const points = live.points.slice(drag.sentPoints);
        drag.sentPoints = live.points.length;
        if (points.length > 0) client.send({ t: 'stroke_chunk', strokeId: drag.strokeId, points });
      }
    }
  };

  const onPointerUp = (): void => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    if (drag.kind === 'move' && drag.layerId && drag.origin) {
      const moved = client.findLayer(drag.layerId);
      if (moved) client.send({ t: 'layer_update', id: drag.layerId, patch: movePatch(moved, drag.origin) });
      return;
    }
    if (drag.kind !== 'stroke' || !drag.strokeId || !drag.liveKey) return;
    const live = client.live.get(drag.liveKey);
    const points = live ? live.points.slice(drag.sentPoints) : [];
    client.send({ t: 'stroke_end', strokeId: drag.strokeId, points });
  };

  // --- keyboard and paste ---------------------------------------------------
  useEffect(() => {
    const isTyping = (target: EventTarget | null): boolean =>
      target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA');

    const down = (e: KeyboardEvent): void => {
      if (e.code === 'Space' && !isTyping(e.target)) {
        spaceRef.current = true;
        e.preventDefault();
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !isTyping(e.target)) {
        e.preventDefault();
        client.send({ t: 'undo' });
      }
    };
    const up = (e: KeyboardEvent): void => {
      if (e.code === 'Space') spaceRef.current = false;
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [client]);

  const onPaste = useCallback(
    async (e: ClipboardEvent): Promise<void> => {
      const target = e.target;
      if (target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      const item = [...(e.clipboardData?.items ?? [])].find((i) => ACCEPTED_PASTE_TYPES.includes(i.type));
      const file = item?.getAsFile();
      if (!file) return;
      e.preventDefault();
      const { blob, width: w, height: h } = await downscaleBlob(file, pasteLimit(client.canvasSize));
      const res = await fetch(`/rooms/${roomId}/images`, { method: 'POST', headers: { 'content-type': 'image/png' }, body: blob });
      if (!res.ok) return;
      const stored = (await res.json()) as { imageId: string };
      const at = pastePlacement({ x: camera.centerX, y: camera.centerY }, { width: w, height: h }, client.canvasSize);
      // Select it and switch to move, so the very next drag moves the paste.
      pastedImageId.current = stored.imageId;
      setTool('move');
      client.send({ t: 'layer_create', layer: { kind: 'reference', imageId: stored.imageId, x: at.x, y: at.y, scale: 1 } });
    },
    [camera.centerX, camera.centerY, client, roomId],
  );

  useEffect(() => {
    const handler = (e: ClipboardEvent): void => void onPaste(e);
    window.addEventListener('paste', handler);
    return () => window.removeEventListener('paste', handler);
  }, [onPaste]);

  // --- render ---------------------------------------------------------------
  /**
   * What this room has actually measured for a profile, falling back to the
   * numbers from docs/experiments/2026-09-05-comfyui/REPORT.md until it has.
   */
  const hint = (p: AIProfileName): string => {
    const ms = client.profileLatency[p] ?? PROFILE_HINT_MS[p];
    return ms >= 10_000 ? `${Math.round(ms / 1000)} s` : `${(ms / 1000).toFixed(1)} s`;
  };

  /**
   * Only mention profiles the backend actually has: on the stream worker there
   * is no quality mode, so "quality ~10 s" would be a promise it cannot keep.
   */
  const profileHint = AI_PROFILES.filter((p) => client.aiProfiles.includes(p))
    .map((p) => `${p} ~${hint(p)}`)
    .join(', ');
  const backendLabel = 'current';

  const statusText =
    client.aiState === 'error'
      ? `AI error: ${client.aiMessage}`
      : client.aiState === 'generating'
        ? 'AI generating...'
        : client.aiState === 'queued'
          ? 'AI queued'
          : client.aiLatencyMs > 0
            ? `AI idle (${client.aiLatencyMs} ms)`
            : 'AI idle';

  const copyInvite = async (): Promise<void> => {
    await navigator.clipboard.writeText(location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="room">
      <header className="topbar">
        <strong>Brush Jam</strong>
        <span className="roomid">room {roomId}</span>
        <button onClick={() => void copyInvite()}>{copied ? 'Copied!' : 'Copy invite URL'}</button>
        <div className="members">
          {client.members.map((m) => (
            <span key={m.userId} className="chip" style={{ borderColor: m.color, color: m.color }}>
              {m.name}
              {m.userId === client.youUserId ? ' (you)' : ''}
            </span>
          ))}
        </div>
        <input
          className="prompt"
          value={promptDraft}
          placeholder="room prompt, e.g. anime style, fantasy town"
          onChange={(e) => {
            setPromptDraft(e.target.value);
            setPromptDirty(true);
          }}
        />
        <div className="segmented" title={profileHint}>
          {AI_PROFILES.map((p) => {
            const supported = client.aiProfiles.includes(p);
            return (
              <button
                key={p}
                className={client.aiProfile === p ? 'active' : ''}
                disabled={!supported}
                title={supported ? undefined : `the ${backendLabel} backend has no ${p} profile`}
                onClick={() => client.send({ t: 'set_ai_settings', aiProfile: p })}
              >
                {p}
              </button>
            );
          })}
        </div>
        <span className="hint">{profileHint}</span>
        <button className={advanced ? 'active' : ''} onClick={() => setAdvanced((v) => !v)}>
          advanced
        </button>
        <span className={`pill ${client.aiState}`}>{statusText}</span>
        <span className={`pill ${client.connected ? 'idle' : 'error'}`}>{client.connected ? 'online' : 'offline'}</span>
      </header>

      {advanced ? (
        <div className="tools">
          <label>
            denoise {Math.min(denoiseDraft ?? client.denoise, client.maxDenoise).toFixed(2)}
            <input
              type="range"
              min={MIN_DENOISE}
              max={client.maxDenoise}
              step={DENOISE_STEP}
              value={Math.min(denoiseDraft ?? client.denoise, client.maxDenoise)}
              onChange={(e) => setDenoiseDraft(Number(e.target.value))}
            />
          </label>
          {client.aiResolutionAdjustable ? (
          <label title="generation resolution; the result is scaled to the canvas">
            AI resolution
            <select
              value={client.aiResolution}
              onChange={(e) => client.send({ t: 'set_ai_settings', aiResolution: Number(e.target.value) })}
            >
              {AI_RESOLUTIONS.filter((r) => r <= client.aiResolutionMax).map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
              {AI_RESOLUTIONS.every((r) => r !== client.aiResolution) ? (
                <option value={client.aiResolution}>{client.aiResolution}</option>
              ) : null}
            </select>
          </label>
          ) : null}
          <input
            className="prompt"
            value={negativeDraft ?? client.negativePrompt}
            maxLength={MAX_NEGATIVE_PROMPT}
            placeholder={DEFAULT_NEGATIVE_PROMPT}
            onChange={(e) => setNegativeDraft(e.target.value)}
          />
        </div>
      ) : null}

      <div className="tools">
        {(['pen', 'noise', 'eraser', 'move'] as const).map((t) => (
          <button key={t} className={tool === t ? 'active' : ''} onClick={() => setTool(t)}>
            {t}
          </button>
        ))}
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
        <label>
          size {width}
          <input type="range" min={1} max={128} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
        </label>
        <button onClick={() => client.send({ t: 'undo' })}>undo (Ctrl+Z)</button>
        <button onClick={fit}>fit</button>
        <button onClick={reset}>100%</button>
        <span className="zoom">{Math.round(camera.zoom * 100)}%</span>
      </div>

      <div className="body">
        <div className="stages">
          <div className="stage-wrap">
            <StageView
              client={client}
              camera={camera}
              kind="human"
              label="Human canvas"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onWheel={onWheel}
            />
          </div>
          <div className="stage-wrap">
            <StageView
              client={client}
              camera={camera}
              kind="ai"
              label="AI canvas"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onWheel={onWheel}
            />
          </div>
        </div>
        <LayerPanel
          client={client}
          layers={layers}
          activeLayerId={activeLayer?.id ?? null}
          maxLayers={MAX_LAYERS}
          onSelect={setActiveLayerId}
        />
      </div>
    </div>
  );
}
