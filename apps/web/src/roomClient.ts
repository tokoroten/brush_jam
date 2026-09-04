import {
  CANVAS_SIZE,
  type AIState,
  type ClientMessage,
  type Layer,
  type Member,
  type Point,
  type Rect,
  type ServerMessage,
  type Stroke,
  type StrokeInit,
} from '@brushjam/shared';
import { createSerialQueue, type SerialQueue } from './serialQueue.js';
import { sessionToken } from './session.js';
import { createRaster, ctxOf, drawStroke, loadImageElement, redrawLayer } from './raster.js';

export interface LiveStroke {
  userId: string;
  init: StrokeInit;
  points: Point[];
}

export interface RemoteCursor {
  x: number;
  y: number;
  at: number;
}

/**
 * One room connection plus every client-side raster. Kept outside React: the
 * component subscribes to a version counter and reads fields directly, so
 * high-frequency drawing never goes through setState.
 */
export class RoomClient {
  readonly aiCanvas = createRaster();
  readonly layerCanvases = new Map<string, HTMLCanvasElement>();
  readonly live = new Map<string, LiveStroke>();
  readonly cursors = new Map<string, RemoteCursor>();
  readonly images = new Map<string, HTMLImageElement>();

  youUserId = '';
  members: Member[] = [];
  layers: Layer[] = [];
  strokes: Stroke[] = [];
  undone = new Set<string>();
  prompt = '';
  humanRevision = 0;
  aiRevision = 0;
  aiState: AIState = 'idle';
  aiMessage = '';
  aiLatencyMs = 0;
  lastCrop: Rect | null = null;
  /** Exact area the server said was authoritative (never hard-coded here). */
  lastApply: Rect | null = null;
  aiWindow = 0;
  aiApply = 0;
  connected = false;

  private socket: WebSocket | null = null;
  private version = 0;
  private listeners = new Set<() => void>();
  private closed = false;
  /**
   * WebSocket frames are ordered, so their handlers must be too. Handling them
   * concurrently let a slow image load apply revision 1 after revision 2.
   */
  private readonly enqueue: SerialQueue = createSerialQueue((err) => console.warn('[brushjam] message handler failed', err));

  constructor(readonly roomId: string, private readonly name: string) {}

  connect(): void {
    if (this.closed) return;
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const token = sessionToken(this.roomId, typeof sessionStorage === 'undefined' ? undefined : sessionStorage);
    const query = `name=${encodeURIComponent(this.name)}&token=${encodeURIComponent(token)}`;
    const socket = new WebSocket(`${protocol}://${location.host}/ws/rooms/${this.roomId}?${query}`);
    this.socket = socket;
    socket.onopen = () => {
      this.connected = true;
      this.bump();
    };
    socket.onclose = () => {
      this.connected = false;
      this.bump();
      if (!this.closed) setTimeout(() => this.connect(), 1000);
    };
    socket.onmessage = (event) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(String(event.data)) as ServerMessage;
      } catch {
        return;
      }
      // Chained, and kept alive across failures, so ordering survives errors.
      this.enqueue(() => this.onMessage(msg));
    };
  }

  dispose(): void {
    this.closed = true;
    this.socket?.close();
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getVersion = (): number => this.version;

  bump(): void {
    this.version += 1;
    for (const l of this.listeners) l();
  }

  send(msg: ClientMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(msg));
  }

  layerCanvas(layerId: string): HTMLCanvasElement {
    let canvas = this.layerCanvases.get(layerId);
    if (!canvas) {
      canvas = createRaster();
      this.layerCanvases.set(layerId, canvas);
    }
    return canvas;
  }

  get orderedLayers(): Layer[] {
    return [...this.layers].sort((a, b) => a.order - b.order);
  }

  findLayer(id: string): Layer | undefined {
    return this.layers.find((l) => l.id === id);
  }

  private async ensureImage(imageId: string): Promise<HTMLImageElement | undefined> {
    const hit = this.images.get(imageId);
    if (hit) return hit;
    try {
      const img = await loadImageElement(`/rooms/${this.roomId}/images/${imageId}`);
      this.images.set(imageId, img);
      return img;
    } catch {
      return undefined;
    }
  }

  private repaint(layer: Layer): void {
    redrawLayer(this.layerCanvas(layer.id), layer, this.strokes, this.undone, this.images);
  }

  private async onMessage(msg: ServerMessage): Promise<void> {
    switch (msg.t) {
      case 'snapshot': {
        const s = msg.snapshot;
        this.youUserId = s.youUserId;
        this.members = s.members;
        this.layers = s.layers;
        this.strokes = s.strokes;
        this.undone = new Set(s.undone);
        this.prompt = s.prompt;
        this.humanRevision = s.humanRevision;
        this.aiRevision = s.aiRevision;
        this.aiState = s.aiState;
        this.aiWindow = s.aiWindow;
        this.aiApply = s.aiApply;
        this.live.clear();
        this.cursors.clear();
        this.layerCanvases.clear();
        for (const layer of s.layers) {
          if (layer.imageId) await this.ensureImage(layer.imageId);
          this.repaint(layer);
        }
        await this.loadAiCanvas();
        break;
      }
      case 'presence': {
        this.members = msg.members;
        // Drop ghosts: cursors and half-drawn strokes from people who left.
        const present = new Set(msg.members.map((m) => m.userId));
        for (const [userId] of this.cursors) if (!present.has(userId)) this.cursors.delete(userId);
        for (const [id, live] of this.live) if (!present.has(live.userId)) this.live.delete(id);
        break;
      }

      case 'stroke_cancel':
        this.live.delete(msg.strokeId);
        break;
      case 'cursor':
        this.cursors.set(msg.userId, { x: msg.x, y: msg.y, at: Date.now() });
        break;
      case 'stroke_start':
        this.live.set(msg.stroke.id, { userId: msg.userId, init: msg.stroke, points: [...msg.stroke.points] });
        break;
      case 'stroke_chunk': {
        const live = this.live.get(msg.strokeId);
        if (live) live.points.push(...msg.points);
        break;
      }
      case 'stroke_end': {
        const live = this.live.get(msg.strokeId);
        if (live) live.points.push(...msg.points);
        break;
      }
      case 'stroke_committed': {
        this.live.delete(msg.stroke.id);
        this.strokes.push(msg.stroke);
        this.humanRevision = msg.humanRevision;
        const layer = this.findLayer(msg.stroke.layerId);
        if (layer) drawStroke(this.layerCanvas(layer.id), msg.stroke);
        break;
      }
      case 'undo_applied': {
        this.undone.add(msg.strokeId);
        this.humanRevision = msg.humanRevision;
        const layer = this.findLayer(msg.layerId);
        if (layer) this.repaint(layer);
        break;
      }
      case 'clear_applied': {
        this.strokes = this.strokes.filter((s) => s.layerId !== msg.layerId);
        this.humanRevision = msg.humanRevision;
        const layer = this.findLayer(msg.layerId);
        if (layer) this.repaint(layer);
        break;
      }
      case 'layer_created':
        this.layers = [...this.layers, msg.layer];
        if (msg.layer.imageId) await this.ensureImage(msg.layer.imageId);
        this.repaint(msg.layer);
        this.humanRevision = msg.humanRevision;
        break;
      case 'layer_updated':
        this.layers = this.layers.map((l) => (l.id === msg.layer.id ? msg.layer : l));
        this.repaint(msg.layer);
        this.humanRevision = msg.humanRevision;
        break;
      case 'layer_deleted':
        this.layers = this.layers.filter((l) => l.id !== msg.id);
        this.strokes = this.strokes.filter((s) => s.layerId !== msg.id);
        this.layerCanvases.delete(msg.id);
        this.humanRevision = msg.humanRevision;
        break;
      case 'layers_reordered':
        this.layers = msg.layers;
        this.humanRevision = msg.humanRevision;
        break;
      case 'prompt_changed':
        this.prompt = msg.prompt;
        break;
      case 'ai_status':
        this.aiState = msg.state;
        this.aiMessage = msg.message ?? '';
        if (msg.latencyMs !== undefined) this.aiLatencyMs = msg.latencyMs;
        break;
      case 'ai_result': {
        // Load first: only advance the AI state once the pixels are really here.
        try {
          const patch = await loadImageElement(msg.url);
          const ctx = ctxOf(this.aiCanvas);
          ctx.clearRect(msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          ctx.drawImage(patch, msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          this.lastCrop = msg.crop;
          this.lastApply = msg.apply;
          this.aiRevision = msg.aiRevision;
          this.aiLatencyMs = msg.latencyMs;
        } catch {
          // A dropped patch would leave a hole, so fall back to the server's
          // authoritative full raster rather than silently losing the update.
          await this.loadAiCanvas();
        }
        break;
      }
      case 'error':
        this.aiMessage = msg.message;
        break;
      default:
        break;
    }
    this.bump();
  }

  private async loadAiCanvas(): Promise<void> {
    try {
      const img = await loadImageElement(`/rooms/${this.roomId}/ai.png?v=${Date.now()}`);
      const ctx = ctxOf(this.aiCanvas);
      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      ctx.drawImage(img, 0, 0);
    } catch {
      /* no AI output yet */
    }
  }
}
