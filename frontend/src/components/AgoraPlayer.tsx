/**
 * AgoraPlayer — live stream viewer via Agora RTC SDK.
 *
 * Fetches a short-lived token from /api/agora/token then joins the channel
 * as a subscriber. The channel name is the camera ID so MediaMTX stream and
 * Agora channel share the same identifier convention.
 *
 * Falls back gracefully when Agora is not configured or the join fails.
 */
import { useEffect, useRef, useState } from "react";
import AgoraRTC, {
  type IAgoraRTCClient,
  type IRemoteVideoTrack,
  type IRemoteAudioTrack,
} from "agora-rtc-sdk-ng";
import { fetchAgoraToken } from "@/services/api";
import { Loader2, AlertCircle, Radio } from "lucide-react";
import { cn } from "@/lib/utils";

interface AgoraPlayerProps {
  /** Camera ID — used as Agora channel name */
  channelName: string;
  className?: string;
  /** Called when stream is active (for status display) */
  onStatusChange?: (status: "connecting" | "live" | "error" | "idle") => void;
}

export function AgoraPlayer({ channelName, className, onStatusChange }: AgoraPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const clientRef = useRef<IAgoraRTCClient | null>(null);
  const [status, setStatus] = useState<"connecting" | "live" | "error" | "idle">("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");

  function updateStatus(s: "connecting" | "live" | "error" | "idle") {
    setStatus(s);
    onStatusChange?.(s);
  }

  useEffect(() => {
    if (!channelName) return;

    let cancelled = false;
    let client: IAgoraRTCClient | null = null;

    async function join() {
      updateStatus("connecting");

      // 1. Fetch short-lived token from our backend
      let tokenResult;
      try {
        tokenResult = await fetchAgoraToken(channelName, 0, "subscriber");
      } catch (e) {
        if (cancelled) return;
        setErrorMsg("Could not fetch Agora token — check OMNI_AGORA_APP_ID / OMNI_AGORA_APP_CERTIFICATE");
        updateStatus("error");
        return;
      }
      if (cancelled) return;

      // 2. Create client
      client = AgoraRTC.createClient({ mode: "live", codec: "h264" });
      clientRef.current = client;

      // 3. Subscribe to remote video/audio
      client.on("user-published", async (user, mediaType) => {
        await client!.subscribe(user, mediaType);
        if (cancelled) return;

        if (mediaType === "video") {
          const track = user.videoTrack as IRemoteVideoTrack;
          if (containerRef.current) {
            track.play(containerRef.current);
          }
          updateStatus("live");
        }
        if (mediaType === "audio") {
          const track = user.audioTrack as IRemoteAudioTrack;
          track.play();
        }
      });

      client.on("user-unpublished", () => {
        if (!cancelled) updateStatus("connecting");
      });

      // 4. Join channel as audience
      try {
        await client.setClientRole("audience");
        await client.join(tokenResult.appId, channelName, tokenResult.token, tokenResult.uid);
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setErrorMsg(`Agora join failed: ${msg}`);
        updateStatus("error");
      }
    }

    void join();

    return () => {
      cancelled = true;
      if (client) {
        client.leave().catch(() => {/* ignore */});
        client.removeAllListeners();
        clientRef.current = null;
      }
      updateStatus("idle");
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channelName]);

  return (
    <div className={cn("relative overflow-hidden rounded-lg bg-black aspect-video", className)}>
      {/* Agora mounts remote video into this div */}
      <div ref={containerRef} className="w-full h-full" />

      {status === "connecting" && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/80">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted/80 gap-2 p-4">
          <AlertCircle className="h-8 w-8 text-destructive shrink-0" />
          <span className="text-xs text-muted-foreground text-center">{errorMsg || "Agora stream unavailable"}</span>
        </div>
      )}

      {status === "live" && (
        <div className="absolute top-2 right-2">
          <span className="flex items-center gap-1 text-[10px] font-mono bg-background/80 text-foreground px-1.5 py-0.5 rounded">
            <Radio className="h-2.5 w-2.5 text-red-500 animate-pulse" />
            AGORA
          </span>
        </div>
      )}
    </div>
  );
}
