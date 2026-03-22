import { useCallback, useEffect, useRef, useState } from "react";
import { useCameraStream } from "@/hooks/useCameraStream";
import { mapBboxToObjectContainPercent } from "@/lib/bboxOverlay";
import { OVERLAY_TRACK_TTL_MS } from "@/lib/overlayConstants";
import type { TrackOverlay } from "@/lib/parseOmniBbox";
import { Video, AlertCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nProvider";


interface LiveStreamPlayerProps {
  hlsUrl?: string;
  webrtcUrl?: string;
  tracks: TrackOverlay[];
  showOverlays: boolean;
  className?: string;
}

export function LiveStreamPlayer({
  hlsUrl,
  webrtcUrl,
  tracks,
  showOverlays,
  className,
}: LiveStreamPlayerProps) {
  const { t } = useI18n();
  const { videoRef, streamStatus } = useCameraStream({ hlsUrl, webrtcUrl });
  const wrapRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ cw: 0, ch: 0, vw: 0, vh: 0 });

  const measure = useCallback(() => {
    const wrap = wrapRef.current;
    const video = videoRef.current;
    if (!wrap || !video) return;
    const cr = wrap.getBoundingClientRect();
    setDims({
      cw: cr.width,
      ch: cr.height,
      vw: video.videoWidth || 1920,
      vh: video.videoHeight || 1080,
    });
  }, [videoRef]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const video = videoRef.current;
    if (!wrap || !video) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(wrap);
    video.addEventListener("loadedmetadata", measure);
    measure();
    return () => {
      ro.disconnect();
      video.removeEventListener("loadedmetadata", measure);
    };
  }, [measure, videoRef, hlsUrl, webrtcUrl]);

  const now = Date.now();
  const fresh = tracks.filter((t) => now - t.lastSeen < OVERLAY_TRACK_TTL_MS);
  const visible = fresh
    .filter((t) => {
      // Only drop detections that are clearly stale (>3s old source timestamp)
      // 1200ms was too strict — caused bbox to flicker/disappear on slow pipelines
      const sourceTs = (t as unknown as { sourceTs?: number }).sourceTs;
      if (typeof sourceTs === "number" && Number.isFinite(sourceTs)) {
        const ageMs = now - sourceTs;
        if (ageMs > 3000) return false;
      }
      return true;
    })
    .filter((t) => (Number.isFinite(t.confidence) ? t.confidence >= 0.20 : true))
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
    .slice(0, 40);

  const colors: Record<string, string> = {
    detection: "#3b82f6",
    vehicle: "#22c55e",
    plate: "#ef4444",
    human: "#a855f7",
  };

  return (
    <div
      ref={wrapRef}
      className={cn("relative bg-muted rounded-lg overflow-hidden aspect-video", className)}
    >
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
      />

      {showOverlays && visible.length > 0 && dims.cw > 0 && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden
        >
          {visible.map((t) => {
            const srcW = t.sourceFrameW && t.sourceFrameW > 0 ? t.sourceFrameW : dims.vw;
            const srcH = t.sourceFrameH && t.sourceFrameH > 0 ? t.sourceFrameH : dims.vh;
            const m = mapBboxToObjectContainPercent(
              t.x1,
              t.y1,
              t.x2,
              t.y2,
              srcW,
              srcH,
              dims.cw,
              dims.ch,
            );
            if (!m) return null;
            const stroke = colors[t.kind] ?? colors.detection;
            const label = t.label.length > 18 ? `${t.label.slice(0, 18)}…` : t.label;
            return (
              <g key={t.id}>
                <rect
                  x={m.x}
                  y={m.y}
                  width={m.w}
                  height={m.h}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                  shapeRendering="crispEdges"
                  strokeLinejoin="round"
                />
                <text
                  x={m.x + 0.6}
                  y={Math.max(m.y + 2.2, 2.2)}
                  fill="#ffffff"
                  stroke="rgba(0,0,0,0.75)"
                  strokeWidth={2.6}
                  vectorEffect="non-scaling-stroke"
                  paintOrder="stroke"
                  fontSize={2.1}
                  className="font-mono"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      )}

      {streamStatus === "connecting" && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/80">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}
      {streamStatus === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/80 gap-2">
          <AlertCircle className="h-8 w-8 text-destructive" />
          <span className="text-sm text-muted-foreground">Stream unavailable</span>
        </div>
      )}
      {streamStatus === "idle" && !hlsUrl && !webrtcUrl && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
          <Video className="h-8 w-8 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">{t("live.noStreamConfigured")}</span>
        </div>
      )}
      {(streamStatus === "webrtc" || streamStatus === "hls") && (
        <div className="absolute top-2 right-2 flex gap-1">
          <span className="text-[10px] font-mono bg-background/80 text-foreground px-1.5 py-0.5 rounded">
            {streamStatus.toUpperCase()}
          </span>
        </div>
      )}
    </div>
  );
}
