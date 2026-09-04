import {
  CANVAS_SIZE,
  MAX_LAYERS,
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

export interface RoomImage {
  id: string;
  mime: string;
  bytes: Buffer;
  width: number;
  height: number;
}

export interface RoomState {
  id: string;
  prompt: string;
  humanRevision: number;
  aiRevision: number;
  layers: Layer[];
  strokes: Stroke[];
  undone: Set<string>;
  members: Map<string, Member>;
  memberSeq: number;
  /** In-progress strokes, keyed by stroke id. */
  pending: Map<string, { userId: string; init: StrokeInit; points: Point[] }>;
  images: Map<string, RoomImage>;
  createdAt: number;
}

export interface ApplyResult {
  /** Messages to send to every member (including the sender). */
  broadcast: ServerMessage[];
  /** Messages to relay to everyone except the sender. */
  relay: ServerMessage[];
  /** World-space rects that changed and should be reconsidered by the AI. */
  dirty: Rect[];
  /** True when the room prompt changed and the AI should re-run. */
  promptChanged?: boolean;
}

const empty = (): ApplyResult => ({ broadcast: [], relay: [], dirty: [] });

export function createRoom(id: string): RoomState {
  return {
    id,
    prompt: 'anime style, fantasy town, vibrant colors',
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
    createdAt: Date.now(),
  };
}

export function addMember(room: RoomState, name: string): Member {
  const member: Member = {
    userId: shortId(10),
    name: name.slice(0, 24) || `artist${room.memberSeq + 1}`,
    color: memberColor(room.memberSeq),
  };
  room.memberSeq += 1;
  room.members.set(member.userId, member);
  return member;
}

export function removeMember(room: RoomState, userId: string): void {
  room.members.delete(userId);
  for (const [strokeId, p] of room.pending) if (p.userId === userId) room.pending.delete(strokeId);
}

export function snapshot(room: RoomState, youUserId: string, aiState: RoomSnapshot['aiState']): RoomSnapshot {
  return {
    roomId: room.id,
    youUserId,
    prompt: room.prompt,
    humanRevision: room.humanRevision,
    aiRevision: room.aiRevision,
    canvasSize: CANVAS_SIZE,
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
      if (!init || typeof init.id !== 'string') return empty();
      const layer = findLayer(room, init.layerId);
      if (!layer || layer.kind !== 'draw' || layer.locked) {
        return empty();
      }
      const clean: StrokeInit = {
        id: init.id.slice(0, 40),
        layerId: layer.id,
        tool: init.tool === 'eraser' ? 'eraser' : 'pen',
        color: isHexColor(init.color) ? init.color : '#000000',
        width: clamp(finite(init.width) ? init.width : 8, 1, 128),
        points: sanitizePoints(init.points),
      };
      room.pending.set(clean.id, { userId, init: clean, points: [...clean.points] });
      return { broadcast: [], relay: [{ t: 'stroke_start', userId, stroke: clean }], dirty: [] };
    }

    case 'stroke_chunk': {
      const p = room.pending.get(msg.strokeId);
      if (!p || p.userId !== userId) return empty();
      const points = sanitizePoints(msg.points);
      p.points.push(...points);
      return { broadcast: [], relay: [{ t: 'stroke_chunk', userId, strokeId: msg.strokeId, points }], dirty: [] };
    }

    case 'stroke_end': {
      const p = room.pending.get(msg.strokeId);
      if (!p || p.userId !== userId) return empty();
      room.pending.delete(msg.strokeId);
      const tail = sanitizePoints(msg.points);
      p.points.push(...tail);
      if (p.points.length === 0) return empty();
      room.humanRevision += 1;
      const stroke: Stroke = {
        id: p.init.id,
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
        relay: [{ t: 'stroke_end', userId, strokeId: stroke.id, points: tail }],
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
      if (!layer) return empty();
      const removed = room.strokes.filter((s) => s.layerId === layer.id);
      if (removed.length === 0 && layer.kind === 'draw') {
        room.humanRevision += 1;
        return { broadcast: [{ t: 'clear_applied', layerId: layer.id, humanRevision: room.humanRevision }], relay: [], dirty: [] };
      }
      room.strokes = room.strokes.filter((s) => s.layerId !== layer.id);
      for (const s of removed) room.undone.delete(s.id);
      room.humanRevision += 1;
      const u = unionRects(removed.filter((s) => !room.undone.has(s.id)).map((s) => s.bbox));
      return {
        broadcast: [{ t: 'clear_applied', layerId: layer.id, humanRevision: room.humanRevision }],
        relay: [],
        dirty: u ? [u] : [],
      };
    }

    case 'layer_create': {
      if (room.layers.length >= MAX_LAYERS) {
        return empty();
      }
      const kind = msg.layer.kind === 'reference' ? 'reference' : 'draw';
      const maxOrder = room.layers.reduce((m, l) => Math.max(m, l.order), -1);
      const image = kind === 'reference' && msg.layer.imageId ? room.images.get(msg.layer.imageId) : undefined;
      if (kind === 'reference' && !image) return empty();
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
      if (!layer) return empty();
      const before = layerDirty(room, layer);
      const patch = msg.patch ?? {};
      if (typeof patch.name === 'string') layer.name = patch.name.slice(0, 32);
      if (typeof patch.visible === 'boolean') layer.visible = patch.visible;
      if (typeof patch.locked === 'boolean') layer.locked = patch.locked;
      if (finite(patch.opacity)) layer.opacity = clamp(patch.opacity, 0, 1);
      if (typeof patch.includeInAI === 'boolean' && layer.kind === 'reference') layer.includeInAI = patch.includeInAI;
      if (layer.kind === 'reference') {
        if (finite(patch.x)) layer.x = patch.x;
        if (finite(patch.y)) layer.y = patch.y;
        if (finite(patch.scale)) layer.scale = clamp(patch.scale, 0.05, 8);
      }
      room.humanRevision += 1;
      const after = layerDirty(room, layer);
      return {
        broadcast: [{ t: 'layer_updated', layer, humanRevision: room.humanRevision }],
        relay: [],
        dirty: layer.includeInAI || layer.kind === 'draw' ? [...before, ...after] : [],
      };
    }

    case 'layer_delete': {
      const layer = findLayer(room, msg.id);
      if (!layer) return empty();
      if (layer.kind === 'draw' && room.layers.filter((l) => l.kind === 'draw').length <= 1) return empty();
      const dirty = layerDirty(room, layer);
      room.layers = room.layers.filter((l) => l.id !== layer.id);
      room.strokes = room.strokes.filter((s) => s.layerId !== layer.id);
      room.humanRevision += 1;
      return {
        broadcast: [{ t: 'layer_deleted', id: layer.id, humanRevision: room.humanRevision }],
        relay: [],
        dirty: layer.includeInAI ? dirty : [],
      };
    }

    case 'layer_reorder': {
      if (!Array.isArray(msg.ids)) return empty();
      const known = msg.ids.filter((id) => findLayer(room, id));
      if (known.length !== room.layers.length) return empty();
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

    case 'set_prompt': {
      const prompt = typeof msg.prompt === 'string' ? msg.prompt.slice(0, 800) : '';
      if (prompt === room.prompt) return empty();
      room.prompt = prompt;
      return { broadcast: [{ t: 'prompt_changed', prompt }], relay: [], dirty: [], promptChanged: true };
    }

    default:
      return empty();
  }
}

/** Strokes intersecting a crop, in log order, skipping undone ones. */
export function strokesForCrop(room: RoomState, crop: Rect, layerId?: string): Stroke[] {
  return room.strokes.filter(
    (s) =>
      !room.undone.has(s.id) &&
      (layerId === undefined || s.layerId === layerId) &&
      s.bbox.x < crop.x + crop.width &&
      crop.x < s.bbox.x + s.bbox.width &&
      s.bbox.y < crop.y + crop.height &&
      crop.y < s.bbox.y + s.bbox.height,
  );
}
