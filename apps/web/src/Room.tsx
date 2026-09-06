import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type JSX } from 'react';
import {
  AI_PROFILES,
  AI_RESOLUTIONS,
  CANVAS_SIZE,
  DEFAULT_NEGATIVE_PROMPT,
  DENOISE_STEP,
  MAX_DENOISE,
  PRESET_GROUPS,
  PROMPT_PRESETS,
  MAX_SEED,
  clampSeed,
  randomSeed,
  MAX_LAYERS,
  MAX_STROKE_ALPHA,
  MIN_STROKE_ALPHA,
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
import {
  brushStorage,
  isSizedTool,
  loadBrushSizes,
  saveBrushSizes,
  sizeForTool,
  alphaForTool,
  hasAlpha,
  loadBrushAlphas,
  saveBrushAlphas,
  withAlpha,
  type BrushAlphas,
  withSize,
  MAX_BRUSH,
  MIN_BRUSH,
  type BrushSizes,
  type SizedTool,
} from './brushSize.js';
import { browserCopyDeps, copyText } from './clipboard.js';
import {
  applyHistorySettings,
  formatLatency,
  formatTime,
  galleryAnnounced,
  galleryFailed,
  galleryLoaded,
  galleryLoading,
  initialGallery,
  selectEntry,
  selectedEntry,
  shouldFetch,
  toggleGallery,
} from './gallery.js';
import { browserDownloadDeps, historyFileName, saveAiImage, saveDrawing } from './save.js';

/**
 * What a pointer press means. Extracted so the one rule that is easy to get
 * wrong is testable: the AI canvas is a view of what the model made, and
 * drawing into it used to accept the stroke, show a preview, and then lose it
 * at the next AI result - which reads as the app eating your work.
 */
export function pointerIntent(input: {
  viewOnly: boolean;
  button: number;
  space: boolean;
  shift: boolean;
  tool: 'pen' | 'eraser' | 'noise' | 'move';
}): 'pan' | 'move' | 'draw' {
  if (input.viewOnly || input.button === 1 || input.space || input.shift) return 'pan';
  return input.tool === 'move' ? 'move' : 'draw';
}

/** How long an action error stays on screen. */
const ACTION_ERROR_MS = 5000;

/**
 * Shown instead of letting someone type into a box the sampler will not read:
 * a distilled 4-step model runs at CFG 1.0, where there is no negative branch.
 */
const NEGATIVE_INACTIVE_HINT = 'inactive with the current fast profile (CFG 1.0)';
import { LayerPanel } from './LayerPanel.js';
import { layerOrigin, layerPoint, movePatch, movedPosition, pickMovableLayer, scaledBy } from './move.js';
import { newId } from './id.js';
import { StageView } from './StageView.js';
import { ACCEPTED_PASTE_TYPES, downscaleBlob, pasteLimit, pastePlacement } from './paste.js';
import { applyRandomPreset, denoiseCeiling, onPresetChange } from './presetPicker.js';
import { RoomClient } from './roomClient.js';
import type { HistoryListing } from '@brushjam/shared';
import { useSharedDraft } from './sharedDraft.js';

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
  // Per tool, so switching back to the noise pen restores the wide brush it
  // needs rather than the line width the pen was left on.
  const [brushSizes, setBrushSizes] = useState<BrushSizes>(() => loadBrushSizes(brushStorage()));
  const [lastSized, setLastSized] = useState<SizedTool>('pen');
  const width = sizeForTool(brushSizes, tool, lastSized);
  const setWidth = (next: number): void => {
    const target = isSizedTool(tool) ? tool : lastSized;
    setBrushSizes((prev) => {
      const updated = withSize(prev, target, next);
      saveBrushSizes(brushStorage(), updated);
      return updated;
    });
  };
  // Opacity is per tool for the same reason: soft shading with a 30% pen must
  // not quietly make the noise pen 30% too, where full strength is the point.
  const [brushAlphas, setBrushAlphas] = useState<BrushAlphas>(() => loadBrushAlphas(brushStorage()));
  const alpha = alphaForTool(brushAlphas, tool, lastSized);
  const setAlpha = (next: number): void => {
    const target = isSizedTool(tool) ? tool : lastSized;
    setBrushAlphas((prev) => {
      const updated = withAlpha(prev, target, next);
      saveBrushAlphas(brushStorage(), updated);
      return updated;
    });
  };
  const [activeLayerId, setActiveLayerId] = useState<string | null>(null);
  // Room-level fields: what is typed here has to survive the round trip to the
  // server and everyone else's edits in the meantime. See sharedDraft.ts.
  // The session epoch is passed to every shared field: a field has to know
  // that the connection it was talking to is gone, or an edit typed during a
  // reconnect is never sent and then overwritten by the snapshot.
  const promptField = useSharedDraft(
    client.prompt,
    (value) => client.send({ t: 'set_prompt', prompt: value }),
    client.sessionEpoch,
  );
  const [copied, setCopied] = useState(false);
  // Read once per render rather than at module scope: jsdom and SSR have no
  // location, and the value has to follow whatever address the page was opened
  // with (a LAN IP, if the invite is to work for anyone else).
  const inviteUrl = typeof location === 'undefined' ? '' : location.href;
  const [advanced, setAdvanced] = useState(false);
  const denoiseField = useSharedDraft(
    client.denoise,
    (value) => client.send({ t: 'set_ai_settings', denoise: value }),
    client.sessionEpoch,
  );
  const negativeField = useSharedDraft(
    client.negativePrompt,
    (value) => client.send({ t: 'set_ai_settings', negativePrompt: value }),
    client.sessionEpoch,
  );
  const seedField = useSharedDraft(
    client.seed,
    (value) => client.send({ t: 'set_ai_settings', seed: value }),
    client.sessionEpoch,
  );
  /** See presetPicker.ts: a one-shot fill of the prompt fields (and, for the
   * presets that carry them, the denoise and profile they want). */
  const presetTarget = {
    prompt: promptField,
    negative: negativeField,
    denoise: denoiseField,
    maxDenoise: client.maxDenoise,
    profiles: client.aiProfiles,
    send: (aiProfile: AIProfileName) => client.send({ t: 'set_ai_settings', aiProfile }),
  };

  /** "use these settings" writes the same fields a preset does, plus the seed. */
  const historyTarget = { ...presetTarget, seed: seedField };

  // --- the history strip ----------------------------------------------------
  const [gallery, setGallery] = useState(initialGallery);
  // Every ai_result carrying a historyN is an announcement that there is
  // something newer to fetch - noted even while the strip is closed, so
  // opening it later is one fetch rather than a fetch and then a correction.
  useEffect(() => {
    setGallery((g) => galleryAnnounced(g, client.latestHistoryN ?? undefined));
  }, [client, client.latestHistoryN, version]);

  // A sequence number rather than a cleanup flag. This effect depends on the
  // gallery state and its own first act is to change it, so a cleanup that
  // cancelled the request would cancel the one it had just started, and the
  // strip would say "loading..." forever. Only a *newer* request cancels an
  // older one.
  const galleryFetch = useRef(0);
  useEffect(() => {
    if (!shouldFetch(gallery)) return;
    const seq = (galleryFetch.current += 1);
    setGallery(galleryLoading);
    void (async () => {
      try {
        const res = await fetch(`/rooms/${roomId}/history`);
        if (!res.ok) throw new Error(`server said ${res.status}`);
        const listing = (await res.json()) as HistoryListing;
        if (galleryFetch.current === seq) setGallery((g) => galleryLoaded(g, listing));
      } catch (err) {
        if (galleryFetch.current === seq) {
          setGallery((g) => galleryFailed(g, err instanceof Error ? err.message : String(err)));
        }
      }
    })();
  }, [gallery, roomId]);

  const shown = selectedEntry(gallery);
  const downloads = useMemo(() => browserDownloadDeps(), []);
  const saveAi = async (): Promise<void> => {
    if (!(await saveAiImage(roomId, client.aiRevision, downloads))) {
      client.noteActionError('nothing to save yet: the AI has not produced a result in this room');
    }
  };
  const saveHuman = async (): Promise<void> => {
    try {
      if (!(await saveDrawing(roomId, client.humanRevision, client, downloads))) {
        client.noteActionError('could not save the drawing');
      }
    } catch (err) {
      client.noteActionError(`could not save the drawing: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

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
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>, viewOnly = false): void => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const screen = { x: e.clientX, y: e.clientY };
    const world = toWorld(e);

    const intent = pointerIntent({ viewOnly, button: e.button, space: spaceRef.current, shift: e.shiftKey, tool });
    if (intent === 'pan') {
      dragRef.current = { kind: 'pan', lastScreen: screen, sentPoints: 0, lastChunkAt: 0 };
      return;
    }
    if (intent === 'move') {
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
    const strokeTool = tool === 'eraser' ? ('eraser' as const) : tool === 'noise' ? ('noise' as const) : ('pen' as const);
    const init = {
      id: liveKey,
      layerId: activeLayer.id,
      tool: strokeTool,
      color,
      width,
      // The eraser removes fully; only pen and noise carry an opacity.
      alpha: strokeTool === 'eraser' ? 1 : alpha,
      points: [point],
    };
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
      // Checked before the upload, not after: the server would accept the
      // image, store it, and then refuse the layer, leaving an orphan.
      if (client.layers.length >= MAX_LAYERS) {
        client.noteActionError(`paste failed: this room already has the maximum of ${MAX_LAYERS} layers`);
        return;
      }
      try {
        const { blob, width: w, height: h } = await downscaleBlob(file, pasteLimit(client.canvasSize));
        const res = await fetch(`/rooms/${roomId}/images`, {
          method: 'POST',
          headers: { 'content-type': 'image/png' },
          body: blob,
        });
        if (!res.ok) {
          const body = (await res.json().catch(() => null)) as { error?: string } | null;
          client.noteActionError(`paste failed: ${body?.error ?? `server said ${res.status}`}`);
          return;
        }
        const stored = (await res.json()) as { imageId: string };
        const at = pastePlacement({ x: camera.centerX, y: camera.centerY }, { width: w, height: h }, client.canvasSize);
        // Select it and switch to move, so the very next drag moves the paste.
        pastedImageId.current = stored.imageId;
        setTool('move');
        client.send({ t: 'layer_create', layer: { kind: 'reference', imageId: stored.imageId, x: at.x, y: at.y, scale: 1 } });
      } catch (err) {
        client.noteActionError(`paste failed: ${err instanceof Error ? err.message : String(err)}`);
      }
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
    const ok = await copyText(location.href, browserCopyDeps());
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      return;
    }
    // The URL is on screen next to the button, so there is always a way.
    client.noteActionError('could not copy automatically - select the URL and copy it');
  };

  // The toast hides itself; the client keeps the message so a late subscriber
  // still sees it, and clearing goes through the client so every view agrees.
  const actionError = client.actionError;
  useEffect(() => {
    if (!actionError) return;
    const timer = setTimeout(() => client.clearActionError(), ACTION_ERROR_MS);
    return () => clearTimeout(timer);
  }, [actionError, client]);

  return (
    <div className="room">
      <header className="topbar">
        <strong>Brush Jam</strong>
        <span className="roomid">room {roomId}</span>
        <button onClick={() => void copyInvite()}>{copied ? 'Copied!' : 'Copy invite URL'}</button>
        <input className="invite" readOnly value={inviteUrl} onFocus={(e) => e.currentTarget.select()} title={inviteUrl} />
        <div className="members">
          {client.members.map((m) => (
            <span key={m.userId} className="chip" style={{ borderColor: m.color, color: m.color }}>
              {m.name}
              {m.userId === client.youUserId ? ' (you)' : ''}
            </span>
          ))}
        </div>
        <div className="prompt-field">
          <input
            className="prompt"
            value={promptField.value}
            placeholder="room prompt, e.g. anime style, fantasy town"
            onChange={(e) => promptField.set(e.target.value)}
            onFocus={promptField.onFocus}
            onBlur={promptField.onBlur}
            onKeyDown={(e) => {
              // Enter means "done": send now and give the field back to the
              // room, so a later change by anyone else lands in it.
              if (e.key !== 'Enter') return;
              promptField.flush();
              e.currentTarget.blur();
            }}
          />
          {promptField.foreign === null ? null : (
            // Someone else changed the room prompt while this one was being
            // typed. Their value is offered, never applied over the cursor.
            <button className="hint foreign" onClick={promptField.adopt}>
              prompt changed by another player: {promptField.foreign || '(empty)'}
            </button>
          )}
        </div>
        <select
          className="preset"
          title="fill the prompt from a preset; it stays editable afterwards"
          value=""
          onChange={(e) => onPresetChange(e, presetTarget)}
        >
          <option value="">preset</option>
          {PRESET_GROUPS.map((group) => (
            <optgroup key={group} label={group}>
              {PROMPT_PRESETS.filter((preset) => preset.group === group).map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <button className="dice" title="random preset" onClick={() => applyRandomPreset(presetTarget)}>
          ⚀
        </button>
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
        <button title="download the AI result as a PNG" onClick={() => void saveAi()}>
          save AI
        </button>
        <button title="download the drawing as a PNG, exactly as it looks here" onClick={() => void saveHuman()}>
          save drawing
        </button>
        <button
          className={gallery.open ? 'active' : ''}
          title="every AI result this room has made"
          onClick={() => setGallery(toggleGallery)}
        >
          history
        </button>
        <button className={advanced ? 'active' : ''} onClick={() => setAdvanced((v) => !v)}>
          advanced
        </button>
        <span className={`pill ${client.aiState}`}>{statusText}</span>
        {client.superseded ? (
          // Not "offline": nothing is wrong with the network, this room is
          // open in another tab. Reconnecting is a deliberate act, because it
          // takes the room back from that tab.
          <button className="pill error" title="this room is open in another tab" onClick={() => client.connect()}>
            opened in another tab - reconnect
          </button>
        ) : (
          <span className={`pill ${client.connected ? 'idle' : 'error'}`}>{client.connected ? 'online' : 'offline'}</span>
        )}
      </header>

      {advanced ? (
        <div className="tools">
          <label>
            {/* The ceiling on the grid the slider actually moves on, so the
                number under the thumb is one the server will accept back. */}
            denoise {Math.min(denoiseField.value, denoiseCeiling(client.maxDenoise)).toFixed(2)}
            <input
              type="range"
              min={MIN_DENOISE}
              max={denoiseCeiling(client.maxDenoise)}
              step={DENOISE_STEP}
              value={Math.min(denoiseField.value, denoiseCeiling(client.maxDenoise))}
              onChange={(e) => denoiseField.set(Number(e.target.value))}
              onPointerUp={denoiseField.flush}
              onBlur={denoiseField.onBlur}
            />
          </label>
          <label title="the room's sampling seed; the same drawing with the same seed comes out the same">
            seed
            <input
              className="seed"
              type="number"
              min={0}
              max={MAX_SEED}
              step={1}
              value={seedField.value}
              onChange={(e) => {
                // A half-typed number is not a seed; ignore it rather than
                // sending a 0 nobody asked for.
                const next = Number(e.target.value);
                if (e.target.value === '' || !Number.isFinite(next)) return;
                seedField.set(clampSeed(next));
              }}
              onFocus={seedField.onFocus}
              onBlur={seedField.onBlur}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return;
                seedField.flush();
                e.currentTarget.blur();
              }}
            />
          </label>
          <button
            className="dice"
            title="random seed"
            onClick={() => {
              // The dice is the whole point of a fixed seed: immediate, no
              // debounce, a different picture from the same drawing.
              seedField.set(randomSeed());
              seedField.flush();
            }}
          >
            ⚀
          </button>
          {seedField.foreign === null ? null : (
            <button className="hint foreign" onClick={seedField.adopt}>
              seed changed by another player: {seedField.foreign}
            </button>
          )}
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
            value={negativeField.value}
            maxLength={MAX_NEGATIVE_PROMPT}
            placeholder={DEFAULT_NEGATIVE_PROMPT}
            disabled={!client.negativePromptActive}
            title={client.negativePromptActive ? undefined : NEGATIVE_INACTIVE_HINT}
            onChange={(e) => negativeField.set(e.target.value)}
            onFocus={negativeField.onFocus}
            onBlur={negativeField.onBlur}
            onKeyDown={(e) => {
              if (e.key !== 'Enter') return;
              negativeField.flush();
              e.currentTarget.blur();
            }}
          />
          {negativeField.foreign === null ? null : (
            <button className="hint foreign" onClick={negativeField.adopt}>
              negative prompt changed by another player: {negativeField.foreign || '(empty)'}
            </button>
          )}
          {client.negativePromptActive ? null : <span className="hint">{NEGATIVE_INACTIVE_HINT}</span>}
        </div>
      ) : null}

      <div className="tools">
        {(['pen', 'noise', 'eraser', 'move'] as const).map((t) => (
          <button
            key={t}
            className={tool === t ? 'active' : ''}
            title={t === 'noise' ? 'noise: the AI invents something here (use a wide brush)' : undefined}
            onClick={() => {
              setTool(t);
              if (isSizedTool(t)) setLastSized(t);
            }}
          >
            {t}
          </button>
        ))}
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} />
        <label>
          size {width}
          <input
            type="range"
            min={MIN_BRUSH}
            max={MAX_BRUSH}
            value={width}
            onChange={(e) => setWidth(Number(e.target.value))}
          />
        </label>
        {hasAlpha(tool) ? (
          <label title="stroke opacity - a stroke that crosses itself stays one strength">
            alpha {Math.round(alpha * 100)}%
            <input
              type="range"
              min={Math.round(MIN_STROKE_ALPHA * 100)}
              max={Math.round(MAX_STROKE_ALPHA * 100)}
              value={Math.round(alpha * 100)}
              onChange={(e) => setAlpha(Number(e.target.value) / 100)}
            />
          </label>
        ) : null}
        <button onClick={() => client.send({ t: 'undo' })}>undo (Ctrl+Z)</button>
        <button onClick={fit}>fit</button>
        <button onClick={reset}>100%</button>
        <span className="zoom">{Math.round(camera.zoom * 100)}%</span>
      </div>

      {gallery.open ? (
        <div className="gallery">
          {!gallery.enabled ? (
            <span className="hint">this server keeps no history (HISTORY_ENABLED=0)</span>
          ) : gallery.error ? (
            <span className="hint">history unavailable: {gallery.error}</span>
          ) : gallery.entries.length === 0 ? (
            <span className="hint">{gallery.loading ? 'loading...' : 'nothing generated in this room yet'}</span>
          ) : (
            <div className="strip">
              {gallery.entries.map((e) => (
                <button
                  key={e.n}
                  className={`thumb ${gallery.selected === e.n ? 'active' : ''}`}
                  title={`#${e.n} ${e.prompt || '(no prompt)'}`}
                  onClick={() => setGallery((g) => selectEntry(g, e.n))}
                >
                  {/* Lazy: a long session's strip is hundreds of JPEGs, and
                      the ones off the end of the row are never looked at. */}
                  <img src={e.url} alt={`result ${e.n}`} loading="lazy" decoding="async" />
                </button>
              ))}
            </div>
          )}
          {shown ? (
            <div className="shot">
              <img src={shown.url} alt={`result ${shown.n}`} />
              <div className="shot-meta">
                <strong>#{shown.n}</strong>
                <span className="hint">{formatTime(shown.time)}</span>
                <span className="prompt-text">{shown.prompt || '(no prompt)'}</span>
                {shown.negativePrompt ? (
                  <span className="hint">negative: {shown.negativePrompt}</span>
                ) : null}
                <span className="hint">
                  {shown.profile} · {shown.aiResolution}px · denoise {shown.denoise} · seed {shown.seed} ·{' '}
                  {formatLatency(shown.latencyMs)}
                </span>
                <div className="shot-actions">
                  <a href={shown.url} download={historyFileName(roomId, shown.n)}>
                    download
                  </a>
                  <button
                    title="put this result's prompt, negative prompt, denoise and seed back into the room"
                    onClick={() => applyHistorySettings(shown, historyTarget)}
                  >
                    use these settings
                  </button>
                  <button onClick={() => setGallery((g) => selectEntry(g, null))}>close</button>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {actionError ? (
        <div className="toast" role="status" onClick={() => client.clearActionError()}>
          {actionError.message}
        </div>
      ) : null}

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
              onPointerDown={(e) => onPointerDown(e, true)}
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
