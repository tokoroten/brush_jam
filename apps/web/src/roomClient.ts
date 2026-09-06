import {
  CLOSE_CAPACITY,
  CLOSE_SUPERSEDED,
  MAX_DENOISE,
  type AIProfileName,
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
import { createRaster, ctxOf, drawStroke, drawStrokeSegment, loadImageElement, redrawLayer } from './raster.js';

/**
 * Injectable seam for the two browser-only things this class touches, so the
 * message-handling logic (ordering, snapshot resets, asset failures) can be
 * exercised in a plain Node test.
 */
export interface ClientDeps {
  createRaster: typeof createRaster;
  loadImage: typeof loadImageElement;
  /** Opens the room socket. Replaced in tests; there is no browser here. */
  openSocket: (url: string) => SocketLike;
}

/** The slice of WebSocket this client uses. */
export interface SocketLike {
  readyState: number;
  onopen: (() => void) | null;
  onclose: ((event?: { code?: number }) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror?: (() => void) | null;
  close(): void;
  send(data: string): void;
}

const browserDeps: ClientDeps = {
  createRaster,
  loadImage: loadImageElement,
  openSocket: (url) => new WebSocket(url) as unknown as SocketLike,
};

/** WebSocket.OPEN, without needing the global to exist. */
const SOCKET_OPEN = 1;
export const RECONNECT_MS = 1000;
/**
 * A full server is not a blip: retrying every second only adds load, so the
 * capacity close backs off from five seconds to a minute.
 */
export const CAPACITY_RECONNECT_MS = 5000;
export const MAX_RECONNECT_MS = 60000;

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
  /** Incremental rasters for in-progress strokes, keyed like `live`. */
  private readonly previews = new Map<string, { canvas: HTMLCanvasElement; drawn: number }>();
  readonly cursors = new Map<string, RemoteCursor>();
  readonly images = new Map<string, HTMLImageElement>();

  youUserId = '';
  members: Member[] = [];
  layers: Layer[] = [];
  strokes: Stroke[] = [];
  undone = new Set<string>();
  prompt = '';
  /** Room-level AI settings, shared like the prompt. */
  denoise = 0.55;
  negativePrompt = '';
  /** The room's sampling seed; re-rolling it is what changes the picture. */
  seed = 0;
  /** Generation size, and the largest this server allows. */
  aiResolution = 1024;
  aiResolutionMax = 1024;
  aiResolutionAdjustable = true;
  aiProfile: AIProfileName = 'fast';
  /** What the server's backend can do; the controls follow this. */
  aiProfiles: AIProfileName[] = ['fast', 'quality'];
  maxDenoise = MAX_DENOISE;
  /**
   * False when the current profile runs at CFG 1.0, where the sampler never
   * evaluates the negative branch: the box would accept text and do nothing.
   */
  negativePromptActive = true;
  /**
   * Whether the picker offers the R18 group. A server setting (PRESETS_R18),
   * so it is false until a snapshot says otherwise: showing it and then taking
   * it away would be worse than showing it a moment late.
   */
  r18Presets = false;
  /**
   * Last measured stroke-to-result time per profile, so the hint under the
   * fast/quality switch reflects this machine rather than my measurements.
   * Not persisted: an empty slot falls back to PROFILE_HINT_MS.
   */
  readonly profileLatency: Partial<Record<AIProfileName, number>> = {};
  humanRevision = 0;
  aiRevision = 0;
  aiState: AIState = 'idle';
  aiMessage = '';
  aiLatencyMs = 0;
  /**
   * The number the server saved the latest result under, or null when it keeps
   * no history. The gallery refreshes on it rather than on every ai_result, so
   * a server with the history off is never asked for a listing.
   */
  latestHistoryN: number | null = null;
  /**
   * Bumped by every snapshot, which is what an admitted connection looks like.
   * Shared fields watch it to resend what a lost socket swallowed, and the
   * capacity backoff resets on it - a refused connection is *opened* and then
   * closed, so `onopen` is not evidence of anything.
   */
  sessionEpoch = 0;
  /**
   * Bumped by every socket this client opens, which is not the same event as
   * being admitted: a socket is open, and accepts writes, for a moment before
   * its snapshot arrives. A write tagged with the connection it went out on
   * can be told apart from one the previous socket swallowed (sharedDraft.ts).
   */
  connectionEpoch = 0;
  lastCrop: Rect | null = null;
  /** Exact area the server said was authoritative (never hard-coded here). */
  lastApply: Rect | null = null;
  aiWindow = 0;
  aiApply = 0;
  /** World canvas size, learned from the snapshot (never hard-coded). */
  canvasSize = 1;
  connected = false;
  /**
   * True once the server replaced this connection with a newer one for the
   * same identity - another tab of the same room. Reconnecting here would
   * evict that tab, which would reconnect and evict this one, forever; so this
   * client stops and the UI offers to take the room back deliberately.
   */
  superseded = false;

  private socket: SocketLike | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  /** Grows while the server keeps saying it is full; reset by a real open. */
  private capacityRetryMs = CAPACITY_RECONNECT_MS;
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
  /**
   * Bumped on every write to `aiCanvas`. A full ai.png load started before a
   * newer patch was painted must not overwrite it when it finally arrives.
   */
  private aiPaintGeneration = 0;

  constructor(
    readonly roomId: string,
    private readonly name: string,
    private readonly deps: ClientDeps = browserDeps,
  ) {
    // The world size is only known once the snapshot arrives; allocating a
    // 4096-square raster up front wasted 64 MB for a 1024 canvas.
    this.aiCanvas = this.deps.createRaster(1);
  }

  /**
   * Idempotent and re-runnable: React StrictMode mounts an effect, tears it
   * down and mounts it again, so connect() after dispose() must produce a live
   * connection rather than a permanently "offline" client.
   */
  connect(): void {
    this.closed = false;
    this.superseded = false;
    this.cancelReconnect();
    this.detach(this.socket);
    this.socket = null;
    this.assets.abort();
    this.assets = new AbortController();
    const protocol = typeof location !== 'undefined' && location.protocol === 'https:' ? 'wss' : 'ws';
    const host = typeof location === 'undefined' ? 'localhost' : location.host;
    const token = sessionToken(this.roomId, typeof sessionStorage === 'undefined' ? undefined : sessionStorage);
    const query = `name=${encodeURIComponent(this.name)}&token=${encodeURIComponent(token)}`;
    const socket = this.deps.openSocket(`${protocol}://${host}/ws/rooms/${this.roomId}?${query}`);
    this.socket = socket;
    this.connectionEpoch += 1;
    socket.onopen = () => {
      this.connected = true;
      // The backoff is NOT reset here. A server with no room for another
      // socket accepts the connection and then closes it with 1013, so every
      // refusal fires onopen: resetting here meant the wait never grew past
      // five seconds however long the server stayed full. It is reset when a
      // snapshot arrives, which is the only evidence of being let in.
      this.bump();
    };
    socket.onclose = (event) => {
      // a socket we already replaced or disposed must not drive state
      if (this.socket !== socket) return;
      this.connected = false;
      const code = event?.code;
      if (code === CLOSE_SUPERSEDED) {
        // Another tab took this identity. Reconnecting would take it straight
        // back and start an eviction loop between the two tabs.
        this.superseded = true;
        this.bump();
        return;
      }
      this.bump();
      if (this.closed) return;
      const delay = code === CLOSE_CAPACITY ? this.capacityRetryMs : RECONNECT_MS;
      if (code === CLOSE_CAPACITY) this.capacityRetryMs = Math.min(MAX_RECONNECT_MS, this.capacityRetryMs * 2);
      this.cancelReconnect();
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.connect();
      }, delay);
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
    this.cancelReconnect();
    this.assets.abort();
    const socket = this.socket;
    this.socket = null;
    this.detach(socket);
    this.connected = false;
  }

  private cancelReconnect(): void {
    if (this.reconnectTimer === null) return;
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  /** Silences a socket before closing it, so its onclose cannot reconnect. */
  private detach(socket: SocketLike | null): void {
    if (!socket) return;
    socket.onopen = null;
    socket.onclose = null;
    socket.onmessage = null;
    if ('onerror' in socket) socket.onerror = null;
    try {
      socket.close();
    } catch {
      /* closing a socket that never opened is fine */
    }
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

  /**
   * Put a message on the wire. False means it did not go: there is no socket,
   * or it is not open yet.
   *
   * The caller has to know. A shared field that treats a dropped message as
   * sent waits forever for an echo that cannot come, and the text typed during
   * a reconnect is simply lost (see sharedDraft.ts).
   */
  send(msg: ClientMessage): boolean {
    if (this.socket?.readyState !== SOCKET_OPEN) return false;
    this.socket.send(JSON.stringify(msg));
    return true;
  }

  layerCanvas(layerId: string): HTMLCanvasElement {
    let canvas = this.layerCanvases.get(layerId);
    if (!canvas) {
      canvas = this.deps.createRaster(this.canvasSize);
      this.layerCanvases.set(layerId, canvas);
    }
    return canvas;
  }

  get orderedLayers(): Layer[] {
    return [...this.layers].sort((a, b) => a.order - b.order);
  }

  /**
   * Something the user tried did not happen: a refused message, a failed
   * paste. Held with a timestamp so the UI can hide it again on its own.
   */
  actionError: { message: string; at: number } | null = null;

  noteActionError(message: string): void {
    this.actionError = { message, at: Date.now() };
    this.bump();
  }

  clearActionError(): void {
    if (!this.actionError) return;
    this.actionError = null;
    this.bump();
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

  /**
   * Drop decoded images no layer references any more. A room where people try
   * several reference photos otherwise holds every one of them, fully decoded,
   * in every browser for the rest of the session - tens of megabytes each.
   * Clearing `src` is what actually lets the browser release the bitmap.
   */
  private pruneImages(): void {
    const wanted = new Set<string>();
    for (const layer of this.layers) if (layer.imageId) wanted.add(layer.imageId);
    for (const [id, img] of this.images) {
      if (wanted.has(id)) continue;
      this.images.delete(id);
      try {
        img.src = '';
      } catch {
        /* a stub image in a test may not accept it; dropping the ref is enough */
      }
    }
    // A load still in flight for a layer that is gone has nothing to paint.
    for (const id of this.loading) if (!wanted.has(id)) this.loading.delete(id);
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

  /** The server owns the world size; adopt it and drop rasters of the old one. */
  private resizeRasters(size: number): void {
    this.canvasSize = size;
    this.aiCanvas.width = size;
    this.aiCanvas.height = size;
    this.layerCanvases.clear();
  }

  /**
   * The raster for an in-progress stroke, extended with whatever points have
   * arrived since the last call. Only the new segment is rendered, which keeps
   * a noise stroke's per-frame cost proportional to the movement, not to the
   * whole stroke.
   */
  previewRaster(id: string): HTMLCanvasElement | null {
    const live = this.live.get(id);
    if (!live || live.points.length === 0) return null;
    let entry = this.previews.get(id);
    if (!entry) {
      entry = { canvas: this.deps.createRaster(this.canvasSize), drawn: 0 };
      this.previews.set(id, entry);
    }
    if (live.points.length > entry.drawn) {
      // overlap by one point so consecutive segments join without a gap
      const from = Math.max(0, entry.drawn - 1);
      drawStrokeSegment(entry.canvas, { ...live.init, points: live.points.slice(from) });
      entry.drawn = live.points.length;
    }
    return entry.canvas;
  }

  private forgetPreview(id: string): void {
    this.previews.delete(id);
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
        // Admitted: the only thing that proves the server had room for us.
        this.capacityRetryMs = CAPACITY_RECONNECT_MS;
        // A new session. Everything sent on the socket this replaces is
        // unacknowledged forever, so the fields are told to reconcile with the
        // room's value and resend whatever it does not have.
        this.sessionEpoch += 1;
        this.youUserId = s.youUserId;
        this.members = s.members;
        this.layers = s.layers;
        this.strokes = s.strokes;
        this.undone = new Set(s.undone);
        this.prompt = s.prompt;
        if (s.canvasSize !== this.canvasSize) this.resizeRasters(s.canvasSize);
        this.denoise = s.denoise;
        this.seed = s.seed;
        this.negativePrompt = s.negativePrompt;
        this.aiResolution = s.aiResolution;
        this.aiResolutionMax = s.aiResolutionMax;
        this.aiResolutionAdjustable = s.aiResolutionAdjustable;
        this.aiProfile = s.aiProfile;
        this.aiProfiles = s.aiProfiles;
        this.maxDenoise = s.maxDenoise;
        this.negativePromptActive = s.negativePromptActive;
        this.r18Presets = s.r18Presets === true;
        this.humanRevision = s.humanRevision;
        this.aiRevision = s.aiRevision;
        this.aiState = s.aiState;
        this.aiWindow = s.aiWindow;
        this.aiApply = s.aiApply;
        this.live.clear();
        this.previews.clear();
        this.cursors.clear();
        this.layerCanvases.clear();
        this.pruneImages();
        for (const layer of s.layers) {
          if (layer.imageId) this.requestImage(layer.imageId);
          this.repaint(layer);
        }
        // A snapshot replaces everything: a restarted or recreated room would
        // otherwise keep showing the previous room's AI pixels.
        ctxOf(this.aiCanvas).clearRect(0, 0, this.canvasSize, this.canvasSize);
        this.aiPaintGeneration += 1;
        this.lastCrop = null;
        this.lastApply = null;
        // Nothing to fetch before the first result (the route 404s by design).
        // aiGeneration, not aiRevision: a generation triggered by a settings
        // change in an untouched room finishes at revision 0, and a joiner that
        // checked the revision stayed blank until somebody drew again.
        if (s.aiGeneration > 0) await this.loadAiCanvas();
        break;
      }
      case 'presence': {
        this.members = msg.members;
        // Drop ghosts: cursors and half-drawn strokes from people who left.
        const present = new Set(msg.members.map((m) => m.userId));
        for (const [userId] of this.cursors) if (!present.has(userId)) this.cursors.delete(userId);
        for (const [id, live] of this.live) {
          if (present.has(live.userId)) continue;
          this.live.delete(id);
          this.forgetPreview(id);
        }
        break;
      }

      case 'stroke_cancel':
        this.live.delete(msg.strokeId);
        this.forgetPreview(msg.strokeId);
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
        this.forgetPreview(msg.stroke.id);
        this.strokes.push(msg.stroke);
        this.humanRevision = msg.humanRevision;
        const layer = this.findLayer(msg.stroke.layerId);
        if (layer) drawStroke(this.layerCanvas(layer.id), msg.stroke, layer);
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
        // A layer can be repointed at a different image; the old one may now
        // be referenced by nobody.
        if (msg.layer.imageId) this.requestImage(msg.layer.imageId);
        this.pruneImages();
        this.repaint(msg.layer);
        this.humanRevision = msg.humanRevision;
        break;
      case 'layer_deleted':
        this.layers = this.layers.filter((l) => l.id !== msg.id);
        this.strokes = this.strokes.filter((s) => s.layerId !== msg.id);
        this.layerCanvases.delete(msg.id);
        this.pruneImages();
        this.humanRevision = msg.humanRevision;
        break;
      case 'layers_reordered':
        this.layers = msg.layers;
        this.humanRevision = msg.humanRevision;
        break;
      case 'ai_capabilities':
        this.aiProfiles = msg.aiProfiles;
        this.maxDenoise = msg.maxDenoise;
        this.aiResolutionMax = msg.aiResolutionMax;
        this.negativePromptActive = msg.negativePromptActive;
        break;
      case 'ai_settings_changed':
        this.denoise = msg.denoise;
        this.seed = msg.seed;
        this.negativePrompt = msg.negativePrompt;
        this.aiResolution = msg.aiResolution;
        this.aiProfile = msg.aiProfile;
        this.negativePromptActive = msg.negativePromptActive;
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
        // Noted before the patch is fetched: the result exists in the history
        // whether or not this client manages to paint it.
        if (typeof msg.historyN === 'number') this.latestHistoryN = msg.historyN;
        // Load first: only advance the AI state once the pixels are really here.
        try {
          // Bounded await: ordering matters here, but a stalled request must
          // never freeze the whole message queue.
          const patch = await this.deps.loadImage(msg.url, { signal: this.assets.signal });
          const ctx = ctxOf(this.aiCanvas);
          ctx.clearRect(msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          ctx.drawImage(patch, msg.rect.x, msg.rect.y, msg.rect.width, msg.rect.height);
          this.aiPaintGeneration += 1;
          this.lastCrop = msg.crop;
          this.lastApply = msg.apply;
          this.aiRevision = msg.aiRevision;
          this.aiLatencyMs = msg.latencyMs;
          // Remember what this profile actually costs on this machine.
          // Attribute to the profile the result was GENERATED with: a switch
          // during an in-flight run would otherwise file a fast timing under
          // quality and leave both estimates wrong.
          this.profileLatency[msg.profile ?? this.aiProfile] = msg.latencyMs;
        } catch {
          // A dropped patch would leave a hole. Recover from the server's
          // authoritative raster, but outside the queue so nothing stalls.
          this.scheduleAiRefresh();
        }
        break;
      }
      case 'error':
        this.aiMessage = msg.message;
        // Also surfaced as a toast: a refusal buried in the AI status line was
        // invisible to whoever caused it.
        this.noteActionError(msg.message);
        break;
      default:
        break;
    }
    this.bump();
  }

  private async loadAiCanvas(): Promise<void> {
    const startedAt = this.aiPaintGeneration;
    try {
      const img = await this.deps.loadImage(`/rooms/${this.roomId}/ai.png?v=${Date.now()}`, { signal: this.assets.signal });
      // Something newer was painted while this full raster was in flight, so
      // it is already stale: drawing it would undo the newer patch.
      if (this.aiPaintGeneration !== startedAt) return;
      const ctx = ctxOf(this.aiCanvas);
      ctx.clearRect(0, 0, this.canvasSize, this.canvasSize);
      ctx.drawImage(img, 0, 0);
      this.aiPaintGeneration += 1;
    } catch {
      /* no AI output yet */
    }
  }
}
