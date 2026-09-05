import type { AIProfileName } from './constants.js';
import type { Point, Rect } from './geometry.js';

export type Tool = 'pen' | 'eraser' | 'noise';
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
  /**
   * Draw layers only: the whole layer is rendered translated by this, while
   * stroke coordinates stay untouched in the log. Moving a layer is not undoable.
   */
  offsetX?: number;
  offsetY?: number;
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
  /**
   * Stroke opacity, 0.05-1. Applied to the stroke as a whole, so a stroke that
   * crosses itself is not darker where it overlaps. Ignored for the eraser,
   * which always removes fully.
   */
  alpha?: number;
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
  /**
   * Accepted AI results so far. Use this, not `aiRevision`, to decide whether
   * there is a full raster worth fetching: a generation started by a settings
   * change in an untouched room lands at revision 0.
   */
  aiGeneration: number;
  canvasSize: number;
  /** Configured AI window / apply sizes, so the client never hard-codes them. */
  aiWindow: number;
  aiApply: number;
  /** Room-level AI settings, shared by everyone like the prompt. */
  denoise: number;
  /**
   * The room's sampling seed. Fixed rather than random per generation, so
   * adding a stroke changes the drawing instead of reshuffling the whole
   * picture; the dice in the UI is how you ask for a different one.
   */
  seed: number;
  /** Speed/quality workflow choice, shared like the prompt. */
  aiProfile: AIProfileName;
  /**
   * Profiles the running backend actually has. The stream worker holds one
   * fused LCM LoRA and offers `fast` only, so the UI must not show `quality`.
   */
  aiProfiles: AIProfileName[];
  /** Largest denoise this backend accepts; the room slider stops here. */
  maxDenoise: number;
  /**
   * Whether the negative prompt does anything with the CURRENT profile. A
   * distilled 4-step model at CFG 1.0 never evaluates the negative branch, so
   * the box is inert and the UI greys it rather than pretending.
   */
  negativePromptActive: boolean;
  /** Empty means "use the built-in default list". */
  negativePrompt: string;
  /** Generation size in px; the result is scaled back to the canvas. */
  aiResolution: number;
  /** Largest resolution this server allows (the configured AI_WINDOW). */
  aiResolutionMax: number;
  /**
   * False in patch mode, where the generation size follows the crop window and
   * the control would be accepted, broadcast and then ignored.
   */
  aiResolutionAdjustable: boolean;
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
  /** Stroke opacity, 0.05-1; absent means fully opaque. */
  alpha?: number;
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
  | {
      t: 'set_ai_settings';
      denoise?: number;
      negativePrompt?: string;
      aiResolution?: number;
      aiProfile?: AIProfileName;
      seed?: number;
    };

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
  | {
      t: 'ai_settings_changed';
      denoise: number;
      negativePrompt: string;
      aiResolution: number;
      aiProfile: AIProfileName;
      negativePromptActive: boolean;
      seed: number;
    }
  /**
   * What the backend can do, when that changes under a live room - a worker
   * restarting smaller, or appearing at all. Without it an open client keeps
   * offering controls the server will now refuse.
   */
  | {
      t: 'ai_capabilities';
      aiProfiles: AIProfileName[];
      maxDenoise: number;
      aiResolutionMax: number;
      negativePromptActive: boolean;
    }
  | { t: 'ai_status'; state: AIState; forRevision: number; message?: string; latencyMs?: number }
  | {
      t: 'ai_result';
      rect: Rect;
      url: string;
      aiRevision: number;
      aiGeneration: number;
      crop: Rect;
      apply: Rect;
      latencyMs: number;
      /** The profile this result was generated with, not the room's current one. */
      profile: AIProfileName;
    }
  | { t: 'error'; message: string };
