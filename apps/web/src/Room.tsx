import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type JSX } from 'react';
import {
  CANVAS_SIZE,
  MAX_LAYERS,
  fitCamera,
  panBy,
  screenToWorld,
  zoomAt,
  type Camera,
  type Layer,
  type Point,
} from '@brushjam/shared';
import { LayerPanel } from './LayerPanel.js';
import { newId } from './id.js';
import { StageView } from './StageView.js';
import { ACCEPTED_PASTE_TYPES, downscaleBlob, pastePlacement } from './paste.js';
import { RoomClient } from './roomClient.js';

export type Tool = 'pen' | 'eraser' | 'move';

const AI_APPLY_HINT = 768;
const CURSOR_INTERVAL_MS = 50;
const CHUNK_INTERVAL_MS = 40;

interface Drag {
  kind: 'stroke' | 'pan' | 'move';
  strokeId?: string;
  layerId?: string;
  lastScreen: { x: number; y: number };
  sentPoints: number;
  lastChunkAt: number;
  origin?: { x: number; y: number };
}

export function Room({ roomId, name }: { roomId: string; name: string }): JSX.Element {
  const client = useMemo(() => new RoomClient(roomId, name), [roomId, name]);
  useSyncExternalStore(client.subscribe, client.getVersion);

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

  const dragRef = useRef<Drag | null>(null);
  const spaceRef = useRef(false);
  const lastCursorAt = useRef(0);

  const layers = client.orderedLayers;
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
    const rect = e.currentTarget.getBoundingClientRect();
    const { w, h } = stageSize(e);
    setCamera((cam) => zoomAt(cam, w, h, e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0015)));
  };

  const fit = (): void => {
    const el = document.querySelector('.stage');
    const rect = el?.getBoundingClientRect();
    setCamera(fitCamera(rect?.width ?? 800, rect?.height ?? 600, CANVAS_SIZE));
  };
  const reset = (): void => setCamera((cam) => ({ ...cam, zoom: 1 }));

  // --- drawing --------------------------------------------------------------
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const screen = { x: e.clientX, y: e.clientY };
    const world = toWorld(e);

    if (e.button === 1 || spaceRef.current || e.shiftKey) {
      dragRef.current = { kind: 'pan', lastScreen: screen, sentPoints: 0, lastChunkAt: 0 };
      return;
    }
    if (tool === 'move' && activeLayer?.kind === 'reference') {
      dragRef.current = {
        kind: 'move',
        layerId: activeLayer.id,
        lastScreen: screen,
        sentPoints: 0,
        lastChunkAt: 0,
        origin: { x: activeLayer.x ?? 0, y: activeLayer.y ?? 0 },
      };
      return;
    }
    if (!activeLayer || activeLayer.kind !== 'draw' || activeLayer.locked) return;

    const strokeId = newId();
    const point: Point = { x: world.x, y: world.y, p: e.pressure > 0 ? e.pressure : 1 };
    const init = { id: strokeId, layerId: activeLayer.id, tool: tool === 'eraser' ? ('eraser' as const) : ('pen' as const), color, width, points: [point] };
    client.live.set(strokeId, { userId: client.youUserId, init, points: [point] });
    client.send({ t: 'stroke_start', stroke: init });
    dragRef.current = { kind: 'stroke', strokeId, lastScreen: screen, sentPoints: 1, lastChunkAt: Date.now() };
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
      const dx = (e.clientX - drag.lastScreen.x) / camera.zoom;
      const dy = (e.clientY - drag.lastScreen.y) / camera.zoom;
      drag.lastScreen = { x: e.clientX, y: e.clientY };
      drag.origin = { x: drag.origin.x + dx, y: drag.origin.y + dy };
      if (now - drag.lastChunkAt > 60) {
        drag.lastChunkAt = now;
        client.send({ t: 'layer_update', id: drag.layerId, patch: { x: drag.origin.x, y: drag.origin.y } });
      }
      return;
    }
    if (drag.kind === 'stroke' && drag.strokeId) {
      const live = client.live.get(drag.strokeId);
      if (!live) return;
      live.points.push({ x: world.x, y: world.y, p: e.pressure > 0 ? e.pressure : 1 });
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
      client.send({ t: 'layer_update', id: drag.layerId, patch: { x: drag.origin.x, y: drag.origin.y } });
      return;
    }
    if (drag.kind !== 'stroke' || !drag.strokeId) return;
    const live = client.live.get(drag.strokeId);
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
      const { blob, width: w, height: h } = await downscaleBlob(file);
      const res = await fetch(`/rooms/${roomId}/images`, { method: 'POST', headers: { 'content-type': 'image/png' }, body: blob });
      if (!res.ok) return;
      const stored = (await res.json()) as { imageId: string };
      const at = pastePlacement({ x: camera.centerX, y: camera.centerY }, { width: w, height: h });
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
        <span className={`pill ${client.aiState}`}>{statusText}</span>
        <span className={`pill ${client.connected ? 'idle' : 'error'}`}>{client.connected ? 'online' : 'offline'}</span>
      </header>

      <div className="tools">
        {(['pen', 'eraser', 'move'] as const).map((t) => (
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
        <button onClick={() => activeLayer && client.send({ t: 'clear_layer', layerId: activeLayer.id })}>clear layer</button>
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
              applySize={AI_APPLY_HINT}
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
              applySize={AI_APPLY_HINT}
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
