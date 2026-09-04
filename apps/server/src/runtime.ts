import type { Rect, ServerMessage } from '@brushjam/shared';
import type { WebSocket } from 'ws';
import type { Config } from './config.js';
import { AIScheduler, type MaskHandle, type RenderJob } from './ai/scheduler.js';
import type { AIBackend } from './ai/backends/index.js';
import { shortId } from './ids.js';
import { checkImage } from './imageInfo.js';
import {
  applyClientMessage,
  captureRenderSnapshot,
  createRoom,
  joinMember,
  removeMember,
  snapshot,
  type RoomState,
} from './room.js';
import { AICanvas, buildMask, forgetImages, renderCropInput } from './raster.js';
import { validateClientMessage } from './validate.js';

const MAX_PATCHES = 24;
/** A slow client is dropped rather than allowed to buffer without limit. */
const MAX_BUFFERED_BYTES = 8 * 1024 * 1024;
export const MAX_IMAGES_PER_ROOM = 32;
export const MAX_IMAGE_BYTES_PER_ROOM = 64 * 1024 * 1024;

export class RoomRuntime {
  readonly state: RoomState;
  readonly scheduler: AIScheduler;
  /** Allocated on the first accepted AI result: a full-size raster costs ~64 MiB. */
  private ai: AICanvas | null = null;
  private readonly sockets = new Map<string, WebSocket>();
  private readonly patches = new Map<string, Buffer>();
  private imageBytes = 0;

  constructor(roomId: string, backend: AIBackend, private readonly config: Config) {
    this.state = createRoom(roomId);
    this.scheduler = new AIScheduler(
      {
        getRevision: () => this.state.humanRevision,
        beginJob: (): RenderJob => {
          // Captured synchronously: the render must not observe later strokes.
          const snap = captureRenderSnapshot(this.state);
          return {
            revision: snap.revision,
            prompt: snap.prompt,
            render: (crop, size) => renderCropInput(snap, crop, size),
          };
        },
        buildMask: (dirty, crop, size, apply): MaskHandle => {
          const built = buildMask(dirty, crop, size, apply);
          return { png: built.png, alpha: built.alpha, empty: built.plan.empty };
        },
        applyResult: async (patch, crop, _apply, mask, forRevision) => {
          const png = await this.aiCanvas().composite(patch, crop, mask as never);
          const id = shortId(10);
          this.patches.set(id, png);
          while (this.patches.size > MAX_PATCHES) {
            const oldest = this.patches.keys().next().value as string | undefined;
            if (oldest === undefined) break;
            this.patches.delete(oldest);
          }
          this.state.aiRevision = forRevision;
          return { rect: crop, url: `/rooms/${roomId}/patches/${id}.png` };
        },
        emit: (msg) => this.broadcast(msg),
      },
      backend,
      {
        window: config.aiWindow,
        apply: config.aiApply,
        steps: config.aiSteps,
        denoise: config.aiDenoise,
        debounceMs: config.aiDebounceMs,
        watchdogMs: config.aiWatchdogMs,
        tag: roomId,
      },
    );
  }

  private aiCanvas(): AICanvas {
    if (!this.ai) this.ai = new AICanvas();
    return this.ai;
  }

  get memberCount(): number {
    return this.sockets.size;
  }

  /** True when the room has been empty long enough to be reclaimed. */
  isIdle(now: number, idleMs: number): boolean {
    return this.sockets.size === 0 && now - this.state.lastActiveAt > idleMs;
  }

  join(socket: WebSocket, name: string, token?: string): string {
    const member = joinMember(this.state, name, token);
    this.sockets.set(member.userId, socket);
    this.send(member.userId, {
      t: 'snapshot',
      snapshot: snapshot(this.state, member.userId, this.scheduler.state, {
        window: this.config.aiWindow,
        apply: this.config.aiApply,
      }),
    });
    this.broadcast({ t: 'presence', members: [...this.state.members.values()] });
    return member.userId;
  }

  leave(userId: string): void {
    for (const cancel of removeMember(this.state, userId)) this.broadcast(cancel);
    this.sockets.delete(userId);
    this.broadcast({ t: 'presence', members: [...this.state.members.values()] });
  }

  handle(userId: string, raw: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      this.send(userId, { t: 'error', message: 'invalid json' });
      return;
    }
    const validated = validateClientMessage(parsed);
    if (!validated.ok) {
      this.send(userId, { t: 'error', message: validated.error });
      return;
    }
    // Belt and braces: a reducer bug must never take the process (and every
    // other room) down from inside a socket event handler.
    try {
      const result = applyClientMessage(this.state, userId, validated.msg);
      for (const msg of result.broadcast) this.broadcast(msg);
      for (const msg of result.relay) this.relay(userId, msg);
      for (const msg of result.toSender ?? []) this.send(userId, msg);
      if (result.dirty.length > 0) this.scheduler.markDirty(result.dirty);
      if (result.promptChanged) this.scheduler.nudge();
    } catch (err) {
      console.error(`[room ${this.state.id}] reducer error on ${validated.msg.t}:`, err);
      this.send(userId, { t: 'error', message: 'the server could not apply that action' });
    }
  }

  async addImage(bytes: Buffer, mime: string): Promise<{ imageId: string; width: number; height: number } | { error: string }> {
    const check = checkImage(bytes, mime);
    if (!check.ok) return { error: check.error };
    if (this.state.images.size >= MAX_IMAGES_PER_ROOM) return { error: 'this room already holds the maximum number of images' };
    if (this.imageBytes + bytes.length > MAX_IMAGE_BYTES_PER_ROOM) return { error: 'this room has reached its image storage limit' };
    const id = shortId(10);
    this.state.images.set(id, { id, mime, bytes, width: check.info.width, height: check.info.height });
    this.imageBytes += bytes.length;
    return { imageId: id, width: check.info.width, height: check.info.height };
  }

  patch(id: string): Buffer | undefined {
    return this.patches.get(id);
  }

  /** False until the first accepted result, so a GET cannot force the allocation. */
  hasAi(): boolean {
    return this.ai !== null;
  }

  aiPng(): Buffer {
    return this.aiCanvas().toPng();
  }

  dispose(): void {
    this.scheduler.stop();
    forgetImages(this.state.images.keys());
    this.state.images.clear();
    this.patches.clear();
    this.ai = null;
  }

  broadcast(msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const [userId, socket] of this.sockets) this.trySend(userId, socket, data);
  }

  private relay(exceptUserId: string, msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const [userId, socket] of this.sockets) if (userId !== exceptUserId) this.trySend(userId, socket, data);
  }

  private send(userId: string, msg: ServerMessage): void {
    const socket = this.sockets.get(userId);
    if (socket) this.trySend(userId, socket, JSON.stringify(msg));
  }

  private trySend(userId: string, socket: WebSocket, data: string): void {
    if (socket.readyState !== 1) return;
    if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
      console.warn(`[room ${this.state.id}] dropping slow client ${userId} (${socket.bufferedAmount} bytes buffered)`);
      socket.terminate();
      return;
    }
    try {
      socket.send(data);
    } catch {
      /* dropped client */
    }
  }
}

export class RoomRegistry {
  private readonly rooms = new Map<string, RoomRuntime>();
  private sweeper: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly backend: AIBackend, private readonly config: Config) {}

  /** Reclaim rooms nobody has been in for a while (each holds a large raster). */
  startSweeper(intervalMs = 60_000): void {
    if (this.sweeper) return;
    this.sweeper = setInterval(() => this.sweep(), intervalMs);
    this.sweeper.unref?.();
  }

  sweep(now = Date.now()): number {
    let removed = 0;
    for (const [id, room] of this.rooms) {
      if (!room.isIdle(now, this.config.roomIdleMs)) continue;
      room.dispose();
      this.rooms.delete(id);
      removed += 1;
    }
    return removed;
  }

  create(): RoomRuntime {
    let id = shortId(8);
    while (this.rooms.has(id)) id = shortId(8);
    const room = new RoomRuntime(id, this.backend, this.config);
    this.rooms.set(id, room);
    return room;
  }

  get(id: string): RoomRuntime | undefined {
    return this.rooms.get(id);
  }

  /** Rooms are created on demand so a shared URL always works. */
  ensure(id: string): RoomRuntime {
    const existing = this.rooms.get(id);
    if (existing) return existing;
    const room = new RoomRuntime(id, this.backend, this.config);
    this.rooms.set(id, room);
    return room;
  }

  dispose(): void {
    if (this.sweeper) clearInterval(this.sweeper);
    this.sweeper = null;
    for (const room of this.rooms.values()) room.dispose();
    this.rooms.clear();
  }

  get size(): number {
    return this.rooms.size;
  }
}

export type { Rect };
