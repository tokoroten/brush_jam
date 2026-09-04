import type { Rect, ServerMessage } from '@brushjam/shared';
import type { WebSocket } from 'ws';
import type { Config } from './config.js';
import { AIScheduler, type MaskHandle } from './ai/scheduler.js';
import type { AIBackend } from './ai/backends/index.js';
import { shortId } from './ids.js';
import {
  addMember,
  applyClientMessage,
  createRoom,
  removeMember,
  snapshot,
  type RoomState,
} from './room.js';
import { AICanvas, buildMask, imageSize, renderCropInput } from './raster.js';

const MAX_PATCHES = 24;

export class RoomRuntime {
  readonly state: RoomState;
  readonly ai = new AICanvas();
  readonly scheduler: AIScheduler;
  private readonly sockets = new Map<string, WebSocket>();
  private readonly patches = new Map<string, Buffer>();

  constructor(roomId: string, backend: AIBackend, private readonly config: Config) {
    this.state = createRoom(roomId);
    this.scheduler = new AIScheduler(
      {
        getPrompt: () => this.state.prompt,
        getRevision: () => this.state.humanRevision,
        renderInput: (crop, size) => renderCropInput(this.state, crop, size),
        buildMask: (dirty, crop, size, applySize): MaskHandle => {
          const built = buildMask(dirty, crop, size, applySize);
          return { png: built.png, alpha: built.alpha, empty: built.plan.empty };
        },
        applyResult: async (patch, crop, mask, forRevision) => {
          const png = await this.ai.composite(patch, crop, mask as never);
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
        tag: roomId,
      },
    );
  }

  get memberCount(): number {
    return this.sockets.size;
  }

  join(socket: WebSocket, name: string): string {
    const member = addMember(this.state, name);
    this.sockets.set(member.userId, socket);
    this.send(member.userId, { t: 'snapshot', snapshot: snapshot(this.state, member.userId, this.scheduler.state) });
    this.broadcast({ t: 'presence', members: [...this.state.members.values()] });
    return member.userId;
  }

  leave(userId: string): void {
    removeMember(this.state, userId);
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
    if (!parsed || typeof parsed !== 'object' || typeof (parsed as { t?: unknown }).t !== 'string') return;
    const result = applyClientMessage(this.state, userId, parsed as never);
    for (const msg of result.broadcast) this.broadcast(msg);
    for (const msg of result.relay) this.relay(userId, msg);
    if (result.dirty.length > 0) this.scheduler.markDirty(result.dirty);
    if (result.promptChanged) this.scheduler.nudge();
  }

  async addImage(bytes: Buffer, mime: string): Promise<{ imageId: string; width: number; height: number }> {
    const { width, height } = await imageSize(bytes);
    const id = shortId(10);
    this.state.images.set(id, { id, mime, bytes, width, height });
    return { imageId: id, width, height };
  }

  patch(id: string): Buffer | undefined {
    return this.patches.get(id);
  }

  aiPng(): Buffer {
    return this.ai.toPng();
  }

  dispose(): void {
    this.scheduler.stop();
  }

  broadcast(msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const socket of this.sockets.values()) trySend(socket, data);
  }

  private relay(exceptUserId: string, msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const [userId, socket] of this.sockets) if (userId !== exceptUserId) trySend(socket, data);
  }

  private send(userId: string, msg: ServerMessage): void {
    const socket = this.sockets.get(userId);
    if (socket) trySend(socket, JSON.stringify(msg));
  }

  /** Exposed for tests. */
  get aiWindow(): number {
    return this.config.aiWindow;
  }
}

function trySend(socket: WebSocket, data: string): void {
  if (socket.readyState === 1) {
    try {
      socket.send(data);
    } catch {
      /* dropped client */
    }
  }
}

export class RoomRegistry {
  private readonly rooms = new Map<string, RoomRuntime>();

  constructor(private readonly backend: AIBackend, private readonly config: Config) {}

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
    for (const room of this.rooms.values()) room.dispose();
    this.rooms.clear();
  }

  get size(): number {
    return this.rooms.size;
  }
}

export type { Rect };
