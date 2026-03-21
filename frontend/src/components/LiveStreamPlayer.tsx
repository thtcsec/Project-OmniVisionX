import { useCallback, useEffect, useRef, useState } from "react";
import { useCameraStream } from "@/hooks/useCameraStream";
import { mapBboxToObjectContainPercent } from "@/lib/bboxOverlay";
import type { TrackOverlay } from "@/lib/parseOmniBbox";
import { Video, AlertCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nProvider";

const STALE_MS = 600;

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
  const fresh = tracks.filter((t) => now - t.lastSeen < STALE_MS);

  const colors: Record<string, string> = {
    detection: "rgba(59, 130, 246, 0.85)",
    vehicle: "rgba(34, 197, 94, 0.85)",
    plate: "rgba(239, 68, 68, 0.9)",
    human: "rgba(168, 85, 247, 0.85)",
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

      {showOverlays && fresh.length > 0 && dims.cw > 0 && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden
        >
          {fresh.map((t) => {
            const m = mapBboxToObjectContainPercent(
              t.x1,
              t.y1,
              t.x2,
              t.y2,
              dims.vw,
              dims.vh,
              dims.cw,
              dims.ch,
            );
            if (!m) return null;
            const stroke = colors[t.kind] ?? colors.detection;
            return (
              <g key={t.id}>
                <rect
                  x={m.x}
                  y={m.y}
                  width={m.w}
                  height={m.h}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={0.35}
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={m.x + 0.2}
                  y={Math.max(m.y - 0.4, 1)}
                  fill={stroke}
                  fontSize={2.2}
                  className="font-mono"
                >
                  {t.label.length > 18 ? `${t.label.slice(0, 18)}…` : t.label}
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
