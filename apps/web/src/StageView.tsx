import { useEffect, useRef, type JSX } from 'react';
import { CANVAS_SIZE, renderStrokes, worldToScreen, type Camera } from '@brushjam/shared';
import type { RoomClient } from './roomClient.js';

export interface StageViewProps {
  client: RoomClient;
  camera: Camera;
  kind: 'human' | 'ai';
  onPointerDown?: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onPointerMove?: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onPointerUp?: (e: React.PointerEvent<HTMLCanvasElement>) => void;
  onWheel?: (e: React.WheelEvent<HTMLCanvasElement>) => void;
  label: string;
}

const CURSOR_TTL_MS = 4000;

/** One viewport onto the shared world. Human and AI stages share one camera. */
export function StageView(props: StageViewProps): JSX.Element {
  const { client, camera, kind, label } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let raf = 0;

    const frame = (): void => {
      raf = requestAnimationFrame(frame);
      const parent = canvas.parentElement;
      if (!parent) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.floor(parent.clientWidth));
      const height = Math.max(1, Math.floor(parent.clientHeight));
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const cam = cameraRef.current;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#22252a';
      ctx.fillRect(0, 0, width, height);

      ctx.save();
      ctx.translate(width / 2 - cam.centerX * cam.zoom, height / 2 - cam.centerY * cam.zoom);
      ctx.scale(cam.zoom, cam.zoom);
      ctx.imageSmoothingEnabled = cam.zoom < 1;

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

      if (kind === 'human') {
        for (const layer of client.orderedLayers) {
          if (!layer.visible) continue;
          const raster = client.layerCanvases.get(layer.id);
          if (!raster) continue;
          ctx.globalAlpha = layer.opacity;
          ctx.drawImage(raster, 0, 0);
        }
        ctx.globalAlpha = 1;
        for (const live of client.live.values()) {
          renderStrokes(ctx as unknown as never, [{ ...live.init, points: live.points }]);
        }
      } else {
        ctx.drawImage(client.aiCanvas, 0, 0);
      }

      if (client.lastCrop) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = Math.max(1, 1 / cam.zoom);
        ctx.strokeStyle = 'rgba(90,160,255,0.55)';
        ctx.strokeRect(client.lastCrop.x, client.lastCrop.y, client.lastCrop.width, client.lastCrop.height);
        // The exact authoritative rect the server reported - never a guess.
        const inner = client.lastApply;
        if (inner) {
          ctx.strokeStyle = 'rgba(90,160,255,0.3)';
          ctx.strokeRect(inner.x, inner.y, inner.width, inner.height);
        }
      }
      ctx.restore();

      // cursors and label live in screen space
      const now = Date.now();
      ctx.font = '12px system-ui, sans-serif';
      for (const [userId, cursor] of client.cursors) {
        if (userId === client.youUserId || now - cursor.at > CURSOR_TTL_MS) continue;
        const member = client.members.find((m) => m.userId === userId);
        if (!member) continue;
        const p = worldToScreen(cam, width, height, cursor.x, cursor.y);
        ctx.fillStyle = member.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(member.name, p.x + 8, p.y - 6);
      }

      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(8, 8, ctx.measureText(label).width + 16, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, 16, 22);
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [client, kind, label]);

  return (
    <canvas
      ref={canvasRef}
      className="stage"
      onPointerDown={props.onPointerDown}
      onPointerMove={props.onPointerMove}
      onPointerUp={props.onPointerUp}
      onPointerCancel={props.onPointerUp}
      onWheel={props.onWheel}
    />
  );
}
