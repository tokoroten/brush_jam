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

/**
 * Injectable seam for the two browser-only things this class touches, so the
 * message-handling logic (ordering, snapshot resets, asset failures) can be
 * exercised in a plain Node test.
 */
export interface ClientDeps {
  createRaster: typeof createRaster;
  loadImage: typeof loadImageElement;
}

const browserDeps: ClientDeps = { createRaster, loadImage: loadImageElement };

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
  readonly aiCanvas: HTMLCanvasElement;
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
  /** Cancels in-flight asset loads on reconnect and dispose. */
  private assets = new AbortController();
  /** Reference images already being fetched, so we never queue two loads. */
  private readonly loading = new Set<string>();
  private aiRefreshQueued = false;

  constructor(
    readonly roomId: string,
    private readonly name: string,
    private readonly deps: ClientDeps = browserDeps,
  ) {
    this.aiCanvas = this.deps.createRaster();
  }

  connect(): void {
    if (this.closed) return;
    this.assets.abort();
    this.assets = new AbortController();
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
    this.assets.abort();
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
      canvas = this.deps.createRaster();
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

  /**
   * Start loading a reference image *without* blocking the ordered message
   * queue. When it arrives the layer is repainted, if it still exists.
   */
  private requestImage(imageId: string): void {
    if (this.images.has(imageId) || this.loading.has(imageId)) return;
    this.loading.add(imageId);
    this.deps
      .loadImage(`/rooms/${this.roomId}/images/${imageId}`, { signal: this.assets.signal })
      .then((img) => {
        this.images.set(imageId, img);
        for (const layer of this.layers) if (layer.imageId === imageId) this.repaint(layer);
        this.bump();
      })
      .catch(() => {
        /* the layer simply stays blank; a later snapshot retries */
      })
      .finally(() => this.loading.delete(imageId));
  }

  /** Re-pull the authoritative AI raster, outside the ordered queue. */
  private scheduleAiRefresh(): void {
    if (this.aiRefreshQueued || this.closed) return;
    this.aiRefreshQueued = true;
    const forRevision = this.aiRevision;
    setTimeout(() => {
      this.aiRefreshQueued = false;
      // only useful if nothing newer has already been painted
      if (this.aiRevision !== forRevision) return;
      void this.loadAiCanvas().then(() => this.bump());
    }, 1000);
  }

  private repaint(layer: Layer): void {
    redrawLayer(this.layerCanvas(layer.id), layer, this.strokes, this.undone, this.images);
  }

  /** Exposed for tests: pushes one message through the ordered queue. */
  receive(msg: ServerMessage): void {
    this.enqueue(() => this.onMessage(msg));
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
          if (layer.imageId) this.requestImage(layer.imageId);
          this.repaint(layer);
        }
        // A snapshot replaces everything: a restarted or recreated room would
        // otherwise keep showing the previous room's AI pixels.
        ctxOf(this.aiCanvas).clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        this.lastCrop = null;
        this.lastApply = null;
        // Nothing to fetch before the first result (the route 404s by design).
        if (s.aiRevision > 0) await this.loadAiCanvas();
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
        if (msg.layer.imageId) this.requestImage(msg.layer.imageId);
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
          // Bounded await: ordering matters here, but a stalled request must
          // never freeze the whole message queue.
          const patch = await this.deps.loadImage(msg.url, { signal: this.assets.signal });
          const ctx = ctxOf(this.aiCanvas);
          ctx.clearRect(msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          ctx.drawImage(patch, msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          this.lastCrop = msg.crop;
          this.lastApply = msg.apply;
          this.aiRevision = msg.aiRevision;
          this.aiLatencyMs = msg.latencyMs;
        } catch {
          // A dropped patch would leave a hole. Recover from the server's
          // authoritative raster, but outside the queue so nothing stalls.
          this.scheduleAiRefresh();
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
      const img = await this.deps.loadImage(`/rooms/${this.roomId}/ai.png?v=${Date.now()}`, { signal: this.assets.signal });
      const ctx = ctxOf(this.aiCanvas);
      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      ctx.drawImage(img, 0, 0);
    } catch {
      /* no AI output yet */
    }
  }
}
