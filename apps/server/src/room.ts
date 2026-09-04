import {
  CANVAS_SIZE,
  DENOISE_STEP,
  MAX_DENOISE,
  MAX_LAYERS,
  MAX_NEGATIVE_PROMPT,
  MIN_DENOISE,
  strokeBBox,
  unionRects,
  type ClientMessage,
  type Layer,
  type Member,
  type Point,
  type Rect,
  type RoomSnapshot,
  type ServerMessage,
  type Stroke,
  type StrokeInit,
} from '@brushjam/shared';
import { memberColor, shortId } from './ids.js';

/** Matches the AI_DENOISE default; a room can be created with the configured one. */
export const DEFAULT_DENOISE = 0.55;

/** Keeps the shared value on the slider's grid and inside its range. */
export function clampDenoise(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_DENOISE;
  const stepped = Math.round(value / DENOISE_STEP) * DENOISE_STEP;
  return Math.min(MAX_DENOISE, Math.max(MIN_DENOISE, Math.round(stepped * 100) / 100));
}

export interface RoomImage {
  id: string;
  mime: string;
  bytes: Buffer;
  width: number;
  height: number;
  /** Upload time, so an image can be swept once nothing references it. */
  createdAt: number;
}

export interface RoomState {
  id: string;
  prompt: string;
  /** img2img strength, adjustable from the Advanced panel. */
  denoise: number;
  /** Empty means "use the backend's built-in default negative list". */
  negativePrompt: string;
  humanRevision: number;
  aiRevision: number;
  layers: Layer[];
  strokes: Stroke[];
  undone: Set<string>;
  members: Map<string, Member>;
  memberSeq: number;
  /** In-progress strokes, keyed by stroke id. */
  pending: Map<string, { userId: string; init: StrokeInit; points: Point[]; startedAt: number; lastActivityAt: number }>;
  images: Map<string, RoomImage>;
  /** Reconnect tokens -> the member identity they own. */
  sessions: Map<string, Member>;
  createdAt: number;
  /** Wall-clock time the room last had a member connected. */
  lastActiveAt: number;
}

export interface ApplyResult {
  /** Messages to send to every member (including the sender). */
  broadcast: ServerMessage[];
  /** Messages to relay to everyone except the sender. */
  relay: ServerMessage[];
  /** World-space rects that changed and should be reconsidered by the AI. */
  dirty: Rect[];
  /** Messages for the sender only (validation feedback). */
  toSender?: ServerMessage[];
  /** True when the room prompt changed and the AI should re-run. */
  promptChanged?: boolean;
}

const empty = (): ApplyResult => ({ broadcast: [], relay: [], dirty: [] });
const refuse = (message: string): ApplyResult => ({ broadcast: [], relay: [], dirty: [], toSender: [{ t: 'error', message }] });

/** Stroke ids are namespaced by author so two clients cannot collide. */
export const qualifyStrokeId = (userId: string, rawId: string): string => `${userId}:${rawId}`;

/** Hard caps on a single in-progress stroke (a client could otherwise stream forever). */
export const MAX_STROKE_POINTS = 50_000;
export const MAX_STROKE_MS = 60_000;
/** A stroke nobody has added points to in this long is abandoned. */
export const PENDING_STROKE_IDLE_MS = 60_000;
/** One person cannot have more than this many strokes in progress at once. */
export const MAX_PENDING_PER_USER = 4;
/** Upper bound on a room's committed stroke log (snapshots are sent in full). */
export const MAX_STROKES_PER_ROOM = 20_000;

/** Sender-only refusal that also clears the sender's optimistic preview. */
const refuseStroke = (userId: string, strokeId: string, message: string): ApplyResult => ({
  broadcast: [],
  relay: [],
  dirty: [],
  toSender: [
    { t: 'stroke_cancel', userId, strokeId, reason: message },
    { t: 'error', message },
  ],
});

export function createRoom(id: string, denoise = DEFAULT_DENOISE): RoomState {
  return {
    id,
    prompt: 'anime style, fantasy town, vibrant colors',
    denoise: clampDenoise(denoise),
    negativePrompt: '',
    humanRevision: 0,
    aiRevision: 0,
    layers: [
      { id: shortId(6), name: 'Layer 1', kind: 'draw', visible: true, locked: false, opacity: 1, order: 0, includeInAI: true },
    ],
    strokes: [],
    undone: new Set(),
    members: new Map(),
    memberSeq: 0,
    pending: new Map(),
    images: new Map(),
    sessions: new Map(),
    createdAt: Date.now(),
    lastActiveAt: Date.now(),
  };
}

const MAX_SESSIONS = 64;

/**
 * Join, resuming a previous identity when the client presents a reconnect token
 * whose member is not currently connected. Without this, a dropped Wi-Fi
 * connection would give the same person a new userId and lose their undo stack.
 */
export function joinMember(room: RoomState, name: string, token?: string): Member {
  if (token) {
    const prior = room.sessions.get(token);
    if (prior) {
      // Resume unconditionally, even if the old socket still looks connected:
      // that is usually a dead TCP connection the server has not noticed. The
      // caller replaces the stale socket. Tokens live in sessionStorage, so two
      // tabs never share one.
      room.members.set(prior.userId, prior);
      room.lastActiveAt = Date.now();
      return prior;
    }
  }
  const member = addMember(room, name);
  if (token) {
    // A new token is only recorded if a slot can be freed without evicting
    // somebody who is still connected; otherwise this participant simply gets a
    // non-resumable identity, which is far better than splitting a live one.
    if (room.sessions.size < MAX_SESSIONS || evictOldestDisconnectedSession(room)) {
      room.sessions.set(token, member);
    }
  }
  return member;
}

/**
 * Make room in the session table without ever dropping a token belonging to
 * someone currently in the room (that would silently split their identity).
 */
function evictOldestDisconnectedSession(room: RoomState): boolean {
  for (const [token, member] of room.sessions) {
    if (room.members.has(member.userId)) continue;
    room.sessions.delete(token);
    return true;
  }
  // Everyone in the table is still connected: keep them all and refuse to
  // record the newcomer's token.
  return false;
}

export function addMember(room: RoomState, name: string): Member {
  const member: Member = {
    userId: shortId(10),
    name: name.slice(0, 24) || `artist${room.memberSeq + 1}`,
    color: memberColor(room.memberSeq),
  };
  room.memberSeq += 1;
  room.members.set(member.userId, member);
  room.lastActiveAt = Date.now();
  return member;
}

/** Drops a member and cancels any stroke they were still drawing. */
export function removeMember(room: RoomState, userId: string): ServerMessage[] {
  room.members.delete(userId);
  room.lastActiveAt = Date.now();
  const cancels: ServerMessage[] = [];
  for (const [strokeId, p] of room.pending) {
    if (p.userId !== userId) continue;
    room.pending.delete(strokeId);
    cancels.push({ t: 'stroke_cancel', userId, strokeId, reason: 'author left' });
  }
  return cancels;
}

/**
 * Abandons strokes nobody has added points to for a while. A client that goes
 * quiet mid-stroke (crashed tab, sleeping laptop) would otherwise leave a live
 * stroke drawn on every other screen forever.
 */
export function expirePendingStrokes(room: RoomState, now = Date.now(), idleMs = PENDING_STROKE_IDLE_MS): ServerMessage[] {
  const cancels: ServerMessage[] = [];
  for (const [strokeId, p] of room.pending) {
    if (now - p.lastActivityAt <= idleMs && now - p.startedAt <= MAX_STROKE_MS) continue;
    room.pending.delete(strokeId);
    cancels.push({ t: 'stroke_cancel', userId: p.userId, strokeId, reason: 'stroke abandoned' });
  }
  return cancels;
}

/** Cancels every pending stroke on a layer (used when the layer disappears). */
function cancelPendingOnLayer(room: RoomState, layerId: string): ServerMessage[] {
  const cancels: ServerMessage[] = [];
  for (const [strokeId, p] of room.pending) {
    if (p.init.layerId !== layerId) continue;
    room.pending.delete(strokeId);
    cancels.push({ t: 'stroke_cancel', userId: p.userId, strokeId, reason: 'layer removed' });
  }
  return cancels;
}

export function snapshot(
  room: RoomState,
  youUserId: string,
  aiState: RoomSnapshot['aiState'],
  ai: { window: number; apply: number },
): RoomSnapshot {
  return {
    roomId: room.id,
    youUserId,
    prompt: room.prompt,
    humanRevision: room.humanRevision,
    aiRevision: room.aiRevision,
    canvasSize: CANVAS_SIZE,
    aiWindow: ai.window,
    aiApply: ai.apply,
    denoise: room.denoise,
    negativePrompt: room.negativePrompt,
    members: [...room.members.values()],
    layers: sortedLayers(room),
    strokes: room.strokes,
    undone: [...room.undone],
    aiState,
  };
}

export const sortedLayers = (room: RoomState): Layer[] => [...room.layers].sort((a, b) => a.order - b.order);

export const findLayer = (room: RoomState, id: string): Layer | undefined => room.layers.find((l) => l.id === id);

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);

function sanitizePoints(raw: unknown): Point[] {
  if (!Array.isArray(raw)) return [];
  const out: Point[] = [];
  for (const p of raw.slice(0, 20000)) {
    if (!p || typeof p !== 'object') continue;
    const { x, y, p: pressure } = p as Point;
    if (!finite(x) || !finite(y)) continue;
    const point: Point = { x: clamp(x, -1024, CANVAS_SIZE + 1024), y: clamp(y, -1024, CANVAS_SIZE + 1024) };
    if (finite(pressure)) point.p = clamp(pressure, 0, 1);
    out.push(point);
  }
  return out;
}

const clamp = (n: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, n));
const isHexColor = (s: unknown): s is string => typeof s === 'string' && /^#[0-9a-fA-F]{6}$/.test(s);

/** Rects touched by a layer's strokes - used when visibility/order/opacity changes. */
function layerDirty(room: RoomState, layer: Layer): Rect[] {
  if (layer.kind === 'reference') {
    const w = (layer.imageWidth ?? 0) * (layer.scale ?? 1);
    const h = (layer.imageHeight ?? 0) * (layer.scale ?? 1);
    if (w <= 0 || h <= 0) return [];
    return [{ x: layer.x ?? 0, y: layer.y ?? 0, width: w, height: h }];
  }
  const boxes = room.strokes.filter((s) => s.layerId === layer.id && !room.undone.has(s.id)).map((s) => s.bbox);
  const u = unionRects(boxes);
  return u ? [u] : [];
}

export function applyClientMessage(room: RoomState, userId: string, msg: ClientMessage): ApplyResult {
  switch (msg.t) {
    case 'cursor': {
      if (!finite(msg.x) || !finite(msg.y)) return empty();
      return { broadcast: [], relay: [{ t: 'cursor', userId, x: msg.x, y: msg.y }], dirty: [] };
    }

    case 'stroke_start': {
      const init = msg.stroke;
      const id = qualifyStrokeId(userId, init.id);
      // Every refusal echoes a stroke_cancel for this id so the sender drops the
      // preview it already started drawing locally.
      const layer = findLayer(room, init.layerId);
      if (!layer || layer.kind !== 'draw' || layer.locked) return refuseStroke(userId, id, 'cannot draw on that layer');
      if (room.pending.has(id) || room.strokes.some((s2) => s2.id === id)) return refuseStroke(userId, id, 'duplicate stroke id');
      let mine = 0;
      for (const p of room.pending.values()) if (p.userId === userId) mine += 1;
      if (mine >= MAX_PENDING_PER_USER) return refuseStroke(userId, id, 'too many strokes in progress');
      if (room.strokes.length >= MAX_STROKES_PER_ROOM) {
        console.warn(`[room ${room.id}] stroke log full (${room.strokes.length}); refusing new strokes`);
        return refuseStroke(userId, id, 'this room has reached its stroke limit');
      }
      const clean: StrokeInit = {
        id,
        layerId: layer.id,
        tool: init.tool,
        color: isHexColor(init.color) ? init.color : '#000000',
        width: clamp(init.width, 1, 128),
        points: sanitizePoints(init.points),
      };
      const now = Date.now();
      room.pending.set(id, { userId, init: clean, points: [...clean.points], startedAt: now, lastActivityAt: now });
      return { broadcast: [], relay: [{ t: 'stroke_start', userId, stroke: clean }], dirty: [] };
    }

    case 'stroke_chunk': {
      const id = qualifyStrokeId(userId, msg.strokeId);
      const p = room.pending.get(id);
      if (!p) return empty();
      const points = sanitizePoints(msg.points);
      if (p.points.length + points.length > MAX_STROKE_POINTS || Date.now() - p.startedAt > MAX_STROKE_MS) {
        room.pending.delete(id);
        return {
          broadcast: [{ t: 'stroke_cancel', userId, strokeId: id, reason: 'stroke too long' }],
          relay: [],
          dirty: [],
          toSender: [{ t: 'error', message: 'stroke exceeded the point or time limit' }],
        };
      }
      p.points.push(...points);
      p.lastActivityAt = Date.now();
      return { broadcast: [], relay: [{ t: 'stroke_chunk', userId, strokeId: id, points }], dirty: [] };
    }

    case 'stroke_end': {
      const id = qualifyStrokeId(userId, msg.strokeId);
      const p = room.pending.get(id);
      if (!p) return empty();
      room.pending.delete(id);
      const layer = findLayer(room, p.init.layerId);
      if (!layer || layer.kind !== 'draw') {
        return { broadcast: [{ t: 'stroke_cancel', userId, strokeId: id, reason: 'layer removed' }], relay: [], dirty: [] };
      }
      // The layer may have been locked while this stroke was being drawn.
      if (layer.locked) {
        return { broadcast: [{ t: 'stroke_cancel', userId, strokeId: id, reason: 'layer locked' }], relay: [], dirty: [] };
      }
      const tail = sanitizePoints(msg.points);
      p.points.push(...tail);
      if (p.points.length === 0 || p.points.length > MAX_STROKE_POINTS) {
        return { broadcast: [{ t: 'stroke_cancel', userId, strokeId: id, reason: 'empty or oversized stroke' }], relay: [], dirty: [] };
      }
      room.humanRevision += 1;
      const stroke: Stroke = {
        id,
        userId,
        layerId: p.init.layerId,
        tool: p.init.tool,
        color: p.init.color,
        width: p.init.width,
        points: p.points,
        revision: room.humanRevision,
        bbox: strokeBBox(p.points, p.init.width),
      };
      room.strokes.push(stroke);
      return {
        broadcast: [{ t: 'stroke_committed', stroke, humanRevision: room.humanRevision }],
        relay: [{ t: 'stroke_end', userId, strokeId: id, points: tail }],
        dirty: [stroke.bbox],
      };
    }

    case 'undo': {
      let target: Stroke | null = null;
      for (let i = room.strokes.length - 1; i >= 0; i--) {
        const s = room.strokes[i]!;
        if (s.userId === userId && !room.undone.has(s.id)) { target = s; break; }
      }
      if (!target) return empty();
      room.undone.add(target.id);
      room.humanRevision += 1;
      return {
        broadcast: [{ t: 'undo_applied', strokeId: target.id, layerId: target.layerId, humanRevision: room.humanRevision }],
        relay: [],
        dirty: [target.bbox],
      };
    }

    case 'clear_layer': {
      const layer = findLayer(room, msg.layerId);
      if (!layer) return refuse('unknown layer');
      const removed = room.strokes.filter((s2) => s2.layerId === layer.id);
      // union of what was actually *visible*, computed before `undone` is mutated
      const visible = removed.filter((s2) => !room.undone.has(s2.id)).map((s2) => s2.bbox);
      const u = unionRects(visible);
      room.strokes = room.strokes.filter((s2) => s2.layerId !== layer.id);
      for (const s2 of removed) room.undone.delete(s2.id);
      const cancels = cancelPendingOnLayer(room, layer.id);
      room.humanRevision += 1;
      return {
        broadcast: [{ t: 'clear_applied', layerId: layer.id, humanRevision: room.humanRevision }, ...cancels],
        relay: [],
        dirty: u ? [u] : [],
      };
    }

    case 'layer_create': {
      if (room.layers.length >= MAX_LAYERS) return refuse('layer limit reached');
      const kind = msg.layer.kind;
      const maxOrder = room.layers.reduce((m, l) => Math.max(m, l.order), -1);
      const image = kind === 'reference' && msg.layer.imageId ? room.images.get(msg.layer.imageId) : undefined;
      if (kind === 'reference' && !image) return refuse('unknown imageId');
      const layer: Layer = {
        id: shortId(6),
        name: (typeof msg.layer.name === 'string' && msg.layer.name.slice(0, 32)) || (kind === 'reference' ? 'Reference' : `Layer ${room.layers.length + 1}`),
        kind,
        visible: true,
        locked: false,
        opacity: 1,
        order: maxOrder + 1,
        includeInAI: kind === 'draw',
      };
      if (image) {
        layer.imageId = image.id;
        layer.imageWidth = image.width;
        layer.imageHeight = image.height;
        layer.x = finite(msg.layer.x) ? msg.layer.x : 0;
        layer.y = finite(msg.layer.y) ? msg.layer.y : 0;
        layer.scale = clamp(finite(msg.layer.scale) ? msg.layer.scale! : 1, 0.05, 8);
      }
      room.layers.push(layer);
      room.humanRevision += 1;
      return {
        broadcast: [{ t: 'layer_created', layer, humanRevision: room.humanRevision }],
        relay: [],
        dirty: layer.includeInAI ? layerDirty(room, layer) : [],
      };
    }

    case 'layer_update': {
      const layer = findLayer(room, msg.id);
      if (!layer) return refuse('unknown layer');
      const patch = msg.patch;
      const wasIncludedInAI = layer.includeInAI;
      const before = layerDirty(room, layer);

      // `name` and `locked` change nothing about the pixels. Renaming a layer
      // must not spend a generation.
      let rendersDifferently = false;
      const setRender = <K extends keyof Layer>(key: K, value: Layer[K]): void => {
        if (layer[key] === value) return;
        layer[key] = value;
        rendersDifferently = true;
      };

      if (typeof patch.name === 'string') layer.name = patch.name.slice(0, 32);
      if (typeof patch.locked === 'boolean') layer.locked = patch.locked;
      if (typeof patch.visible === 'boolean') setRender('visible', patch.visible);
      if (finite(patch.opacity)) setRender('opacity', clamp(patch.opacity, 0, 1));
      if (typeof patch.includeInAI === 'boolean' && layer.kind === 'reference') setRender('includeInAI', patch.includeInAI);
      if (layer.kind === 'reference') {
        if (finite(patch.x)) setRender('x', patch.x);
        if (finite(patch.y)) setRender('y', patch.y);
        if (finite(patch.scale)) setRender('scale', clamp(patch.scale, 0.05, 8));
      }

      room.humanRevision += 1;
      const after = layerDirty(room, layer);
      // Turning "AI input" off still has to repaint where the layer used to be.
      const affectsAI = rendersDifferently && (wasIncludedInAI || layer.includeInAI);
      return {
        broadcast: [{ t: 'layer_updated', layer, humanRevision: room.humanRevision }],
        relay: [],
        dirty: affectsAI ? [...before, ...after] : [],
      };
    }

    case 'layer_delete': {
      const layer = findLayer(room, msg.id);
      if (!layer) return refuse('unknown layer');
      if (layer.kind === 'draw' && room.layers.filter((l) => l.kind === 'draw').length <= 1) {
        return refuse('cannot delete the last draw layer');
      }
      const dirty = layerDirty(room, layer);
      const cancels = cancelPendingOnLayer(room, layer.id);
      room.layers = room.layers.filter((l) => l.id !== layer.id);
      for (const s2 of room.strokes) if (s2.layerId === layer.id) room.undone.delete(s2.id);
      room.strokes = room.strokes.filter((s2) => s2.layerId !== layer.id);
      room.humanRevision += 1;
      return {
        broadcast: [{ t: 'layer_deleted', id: layer.id, humanRevision: room.humanRevision }, ...cancels],
        relay: [],
        dirty: layer.includeInAI ? dirty : [],
      };
    }

    case 'layer_reorder': {
      const known = msg.ids.filter((id) => findLayer(room, id));
      if (known.length !== room.layers.length) return refuse('layer_reorder must list every layer exactly once');
      known.forEach((id, index) => { findLayer(room, id)!.order = index; });
      room.humanRevision += 1;
      const dirty = room.layers.filter((l) => l.includeInAI).flatMap((l) => layerDirty(room, l));
      const u = unionRects(dirty);
      return {
        broadcast: [{ t: 'layers_reordered', layers: sortedLayers(room), humanRevision: room.humanRevision }],
        relay: [],
        dirty: u ? [u] : [],
      };
    }

    case 'set_ai_settings': {
      // Both are room-level, so a change behaves exactly like a prompt change:
      // everyone sees it, and the AI re-runs without anyone having to draw.
      const denoise = msg.denoise === undefined ? room.denoise : clampDenoise(msg.denoise);
      const negativePrompt = msg.negativePrompt === undefined ? room.negativePrompt : msg.negativePrompt.slice(0, MAX_NEGATIVE_PROMPT);
      if (denoise === room.denoise && negativePrompt === room.negativePrompt) return empty();
      room.denoise = denoise;
      room.negativePrompt = negativePrompt;
      return { broadcast: [{ t: 'ai_settings_changed', denoise, negativePrompt }], relay: [], dirty: [], promptChanged: true };
    }

    case 'set_prompt': {
      const prompt = msg.prompt.slice(0, 800);
      if (prompt === room.prompt) return empty();
      room.prompt = prompt;
      return { broadcast: [{ t: 'prompt_changed', prompt }], relay: [], dirty: [], promptChanged: true };
    }

    default:
      return empty();
  }
}

/**
 * An immutable view of everything a render needs, captured synchronously so an
 * AI request cannot mix state from two different revisions while it awaits.
 */
export interface RenderSnapshot {
  revision: number;
  prompt: string;
  denoise: number;
  negativePrompt: string;
  layers: Layer[];
  strokes: Stroke[];
  undone: ReadonlySet<string>;
  images: Map<string, RoomImage>;
}

export function captureRenderSnapshot(room: RoomState): RenderSnapshot {
  return {
    revision: room.humanRevision,
    prompt: room.prompt,
    denoise: room.denoise,
    negativePrompt: room.negativePrompt,
    layers: sortedLayers(room).map((l) => ({ ...l })),
    strokes: [...room.strokes],
    undone: new Set(room.undone),
    images: new Map(room.images),
  };
}

/** Strokes intersecting a crop, in log order, skipping undone ones. */
export function strokesForCrop(
  source: { strokes: readonly Stroke[]; undone: ReadonlySet<string> },
  crop: Rect,
  layerId?: string,
): Stroke[] {
  return source.strokes.filter(
    (s) =>
      !source.undone.has(s.id) &&
      (layerId === undefined || s.layerId === layerId) &&
      s.bbox.x < crop.x + crop.width &&
      crop.x < s.bbox.x + s.bbox.width &&
      s.bbox.y < crop.y + crop.height &&
      crop.y < s.bbox.y + s.bbox.height,
  );
}
