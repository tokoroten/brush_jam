import { MAX_AI_RESOLUTION, MAX_DENOISE, MAX_NEGATIVE_PROMPT, MIN_AI_RESOLUTION, MIN_DENOISE, type ClientMessage, type Layer, type LayerKind, type Point } from '@brushjam/shared';

/**
 * Hand-written runtime validation for every client message. The reducer is
 * allowed to assume well-formed input; anything that fails here is answered with
 * an `error` frame instead of throwing inside the socket handler (a thrown
 * exception there would take the whole process, and every other room, down).
 */
export type ValidationResult = { ok: true; msg: ClientMessage } | { ok: false; error: string };

const bad = (error: string): ValidationResult => ({ ok: false, error });
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === 'string';
const isId = (v: unknown): v is string => isStr(v) && v.length > 0 && v.length <= 64;

export const MAX_POINTS_PER_MESSAGE = 20_000;

function points(value: unknown): Point[] | null {
  if (!Array.isArray(value)) return null;
  if (value.length > MAX_POINTS_PER_MESSAGE) return null;
  const out: Point[] = [];
  for (const raw of value) {
    if (!isObj(raw) || !isNum(raw.x) || !isNum(raw.y)) return null;
    const p: Point = { x: raw.x, y: raw.y };
    if (raw.p !== undefined) {
      if (!isNum(raw.p)) return null;
      p.p = raw.p;
    }
    out.push(p);
  }
  return out;
}

const LAYER_PATCH_KEYS = new Set(['name', 'visible', 'locked', 'opacity', 'includeInAI', 'x', 'y', 'scale', 'offsetX', 'offsetY']);

function layerPatch(value: unknown): Record<string, unknown> | null {
  if (!isObj(value)) return null;
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(value)) {
    if (!LAYER_PATCH_KEYS.has(key)) continue;
    if (key === 'name' && !isStr(v)) return null;
    if ((key === 'visible' || key === 'locked' || key === 'includeInAI') && typeof v !== 'boolean') return null;
    if ((key === 'opacity' || key === 'x' || key === 'y' || key === 'scale' || key === 'offsetX' || key === 'offsetY') && !isNum(v)) {
      return null;
    }
    out[key] = v;
  }
  return out;
}

export function validateClientMessage(raw: unknown): ValidationResult {
  if (!isObj(raw)) return bad('message must be an object');
  const t = raw.t;
  if (!isStr(t)) return bad('missing message type');

  switch (t) {
    case 'cursor':
      if (!isNum(raw.x) || !isNum(raw.y)) return bad('cursor needs finite x and y');
      return { ok: true, msg: { t: 'cursor', x: raw.x, y: raw.y } };

    case 'stroke_start': {
      const s = raw.stroke;
      if (!isObj(s)) return bad('stroke_start needs a stroke object');
      if (!isId(s.id)) return bad('stroke needs an id');
      if (!isId(s.layerId)) return bad('stroke needs a layerId');
      if (s.tool !== 'pen' && s.tool !== 'eraser' && s.tool !== 'noise') return bad('stroke tool must be pen, eraser or noise');
      if (!isStr(s.color)) return bad('stroke needs a color');
      if (!isNum(s.width)) return bad('stroke needs a finite width');
      const pts = points(s.points);
      if (!pts) return bad('stroke points are malformed or too many');
      return {
        ok: true,
        msg: { t: 'stroke_start', stroke: { id: s.id, layerId: s.layerId, tool: s.tool, color: s.color, width: s.width, points: pts } },
      };
    }

    case 'stroke_chunk':
    case 'stroke_end': {
      if (!isId(raw.strokeId)) return bad(`${t} needs a strokeId`);
      const pts = points(raw.points);
      if (!pts) return bad(`${t} points are malformed or too many`);
      return { ok: true, msg: { t, strokeId: raw.strokeId, points: pts } };
    }

    case 'undo':
      return { ok: true, msg: { t: 'undo' } };

    case 'clear_layer':
      if (!isId(raw.layerId)) return bad('clear_layer needs a layerId');
      return { ok: true, msg: { t: 'clear_layer', layerId: raw.layerId } };

    case 'layer_create': {
      const l = raw.layer;
      if (!isObj(l)) return bad('layer_create needs a layer object');
      if (l.kind !== 'draw' && l.kind !== 'reference') return bad('layer kind must be draw or reference');
      const patch = layerPatch(l);
      if (!patch) return bad('layer fields are malformed');
      if (l.imageId !== undefined && !isId(l.imageId)) return bad('imageId must be a string');
      const layer: Partial<Layer> & { kind: LayerKind } = { ...(patch as Partial<Layer>), kind: l.kind };
      if (isId(l.imageId)) layer.imageId = l.imageId;
      return { ok: true, msg: { t: 'layer_create', layer } };
    }

    case 'layer_update': {
      if (!isId(raw.id)) return bad('layer_update needs an id');
      const patch = layerPatch(raw.patch);
      if (!patch) return bad('layer patch is malformed');
      return { ok: true, msg: { t: 'layer_update', id: raw.id, patch } };
    }

    case 'layer_delete':
      if (!isId(raw.id)) return bad('layer_delete needs an id');
      return { ok: true, msg: { t: 'layer_delete', id: raw.id } };

    case 'layer_reorder': {
      const ids = raw.ids;
      if (!Array.isArray(ids) || ids.length === 0 || ids.length > 64 || !ids.every(isId)) return bad('layer_reorder needs an id list');
      if (new Set(ids as string[]).size !== ids.length) return bad('layer_reorder ids must be unique');
      return { ok: true, msg: { t: 'layer_reorder', ids: ids as string[] } };
    }

    case 'set_prompt':
      if (!isStr(raw.prompt)) return bad('set_prompt needs a string prompt');
      return { ok: true, msg: { t: 'set_prompt', prompt: raw.prompt } };

    case 'set_ai_settings': {
      const msg: ClientMessage = { t: 'set_ai_settings' };
      if (raw.denoise !== undefined) {
        if (!isNum(raw.denoise)) return bad('set_ai_settings denoise must be a finite number');
        if (raw.denoise < MIN_DENOISE || raw.denoise > MAX_DENOISE) {
          return bad(`set_ai_settings denoise must be between ${MIN_DENOISE} and ${MAX_DENOISE}`);
        }
        msg.denoise = raw.denoise;
      }
      if (raw.negativePrompt !== undefined) {
        if (!isStr(raw.negativePrompt)) return bad('set_ai_settings negativePrompt must be a string');
        if (raw.negativePrompt.length > MAX_NEGATIVE_PROMPT) return bad('set_ai_settings negativePrompt is too long');
        msg.negativePrompt = raw.negativePrompt;
      }
      if (raw.aiResolution !== undefined) {
        if (!isNum(raw.aiResolution)) return bad('set_ai_settings aiResolution must be a finite number');
        if (raw.aiResolution < MIN_AI_RESOLUTION || raw.aiResolution > MAX_AI_RESOLUTION) {
          return bad(`set_ai_settings aiResolution must be between ${MIN_AI_RESOLUTION} and ${MAX_AI_RESOLUTION}`);
        }
        if (raw.aiResolution % 64 !== 0) return bad('set_ai_settings aiResolution must be a multiple of 64');
        msg.aiResolution = raw.aiResolution;
      }
      if (msg.denoise === undefined && msg.negativePrompt === undefined && msg.aiResolution === undefined) {
        return bad('set_ai_settings needs denoise, negativePrompt or aiResolution');
      }
      return { ok: true, msg };
    }

    default:
      return bad(`unknown message type: ${t.slice(0, 32)}`);
  }
}
