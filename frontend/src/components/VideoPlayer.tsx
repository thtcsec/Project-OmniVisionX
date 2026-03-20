import { useCameraStream } from "@/hooks/useCameraStream";
import { Video, AlertCircle, Loader2 } from "lucide-react";

interface VideoPlayerProps {
  hlsUrl?: string;
  webrtcUrl?: string;
  className?: string;
}

export function VideoPlayer({ hlsUrl, webrtcUrl, className }: VideoPlayerProps) {
  const { videoRef, streamStatus } = useCameraStream({ hlsUrl, webrtcUrl });

  return (
    <div className={`relative bg-muted rounded-lg overflow-hidden aspect-video ${className ?? ""}`}>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-contain"
      />
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
          <span className="text-sm text-muted-foreground">No stream configured</span>
        </div>
      )}
      {(streamStatus === "webrtc" || streamStatus === "hls") && (
        <div className="absolute top-2 right-2">
          <span className="text-[10px] font-mono bg-background/80 text-foreground px-1.5 py-0.5 rounded">
            {streamStatus.toUpperCase()}
          </span>
        </div>
      )}
    </div>
  );
}
