import type { Point, Rect } from './geometry.js';

export type Tool = 'pen' | 'eraser';
export type LayerKind = 'draw' | 'reference';

export interface Layer {
  id: string;
  name: string;
  kind: LayerKind;
  visible: boolean;
  locked: boolean;
  /** 0..1 */
  opacity: number;
  /** Smaller = further back. Rendered ascending. */
  order: number;
  /** Draw layers are always AI input; reference layers default to false. */
  includeInAI: boolean;
  /** reference layers only */
  imageId?: string;
  x?: number;
  y?: number;
  scale?: number;
  imageWidth?: number;
  imageHeight?: number;
}

export interface Stroke {
  id: string;
  userId: string;
  layerId: string;
  tool: Tool;
  color: string;
  width: number;
  points: Point[];
  /** humanRevision at which this stroke was committed. */
  revision: number;
  bbox: Rect;
}

export interface Member {
  userId: string;
  name: string;
  color: string;
}

export type AIState = 'idle' | 'queued' | 'generating' | 'error';

export interface RoomSnapshot {
  roomId: string;
  youUserId: string;
  prompt: string;
  humanRevision: number;
  aiRevision: number;
  canvasSize: number;
  /** Configured AI window / apply sizes, so the client never hard-codes them. */
  aiWindow: number;
  aiApply: number;
  /** Room-level AI settings, shared by everyone like the prompt. */
  denoise: number;
  /** Empty means "use the built-in default list". */
  negativePrompt: string;
  members: Member[];
  layers: Layer[];
  strokes: Stroke[];
  undone: string[];
  aiState: AIState;
}

/** Stroke data as sent by a client (server assigns userId/revision/bbox). */
export interface StrokeInit {
  id: string;
  layerId: string;
  tool: Tool;
  color: string;
  width: number;
  points: Point[];
}

export type ClientMessage =
  | { t: 'cursor'; x: number; y: number }
  | { t: 'stroke_start'; stroke: StrokeInit }
  | { t: 'stroke_chunk'; strokeId: string; points: Point[] }
  | { t: 'stroke_end'; strokeId: string; points: Point[] }
  | { t: 'undo' }
  | { t: 'clear_layer'; layerId: string }
  | { t: 'layer_create'; layer: Partial<Layer> & { kind: LayerKind } }
  | { t: 'layer_update'; id: string; patch: Partial<Layer> }
  | { t: 'layer_delete'; id: string }
  | { t: 'layer_reorder'; ids: string[] }
  | { t: 'set_prompt'; prompt: string }
  | { t: 'set_ai_settings'; denoise?: number; negativePrompt?: string };

export type ServerMessage =
  | { t: 'snapshot'; snapshot: RoomSnapshot }
  | { t: 'presence'; members: Member[] }
  | { t: 'cursor'; userId: string; x: number; y: number }
  | { t: 'stroke_start'; userId: string; stroke: StrokeInit }
  | { t: 'stroke_chunk'; userId: string; strokeId: string; points: Point[] }
  | { t: 'stroke_end'; userId: string; strokeId: string; points: Point[] }
  | { t: 'stroke_committed'; stroke: Stroke; humanRevision: number }
  | { t: 'stroke_cancel'; userId: string; strokeId: string; reason: string }
  | { t: 'undo_applied'; strokeId: string; layerId: string; humanRevision: number }
  | { t: 'clear_applied'; layerId: string; humanRevision: number }
  | { t: 'layer_created'; layer: Layer; humanRevision: number }
  | { t: 'layer_updated'; layer: Layer; humanRevision: number }
  | { t: 'layer_deleted'; id: string; humanRevision: number }
  | { t: 'layers_reordered'; layers: Layer[]; humanRevision: number }
  | { t: 'prompt_changed'; prompt: string }
  | { t: 'ai_settings_changed'; denoise: number; negativePrompt: string }
  | { t: 'ai_status'; state: AIState; forRevision: number; message?: string; latencyMs?: number }
  | { t: 'ai_result'; rect: Rect; url: string; aiRevision: number; crop: Rect; apply: Rect; latencyMs: number }
  | { t: 'error'; message: string };
