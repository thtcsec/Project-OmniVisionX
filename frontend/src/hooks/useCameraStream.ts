import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";

type StreamStatus = "idle" | "connecting" | "webrtc" | "hls" | "error";

interface UseCameraStreamOptions {
  hlsUrl?: string;
  webrtcUrl?: string;
}

export function useCameraStream({ hlsUrl, webrtcUrl }: UseCameraStreamOptions) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let cancelled = false;

    async function tryWebRTC() {
      if (!webrtcUrl || cancelled) return false;
      try {
        setStreamStatus("connecting");
        const pc = new RTCPeerConnection();
        pcRef.current = pc;
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.addTransceiver("audio", { direction: "recvonly" });
        pc.ontrack = (e) => {
          if (video) video.srcObject = e.streams[0];
        };
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        const res = await fetch(webrtcUrl, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: offer.sdp,
        });
        if (!res.ok) throw new Error("WHEP failed");
        const answer = await res.text();
        await pc.setRemoteDescription({ type: "answer", sdp: answer });
        if (!cancelled) setStreamStatus("webrtc");
        return true;
      } catch {
        pcRef.current?.close();
        pcRef.current = null;
        return false;
      }
    }

    function tryHLS() {
      if (!hlsUrl || cancelled) return false;
      setStreamStatus("connecting");
      if (Hls.isSupported()) {
        const hls = new Hls({ enableWorker: true, lowLatencyMode: true });
        hlsRef.current = hls;
        hls.loadSource(hlsUrl);
        hls.attachMedia(video!);
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
          video?.play().catch(() => {});
          if (!cancelled) setStreamStatus("hls");
        });
        hls.on(Hls.Events.ERROR, (_e, data) => {
          if (data.fatal && !cancelled) setStreamStatus("error");
        });
        return true;
      } else if (video?.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = hlsUrl;
        video.addEventListener("loadedmetadata", () => {
          video.play().catch(() => {});
          if (!cancelled) setStreamStatus("hls");
        });
        return true;
      }
      return false;
    }

    (async () => {
      const webrtcOk = await tryWebRTC();
      if (!webrtcOk && !cancelled) {
        const hlsOk = tryHLS();
        if (!hlsOk && !cancelled) setStreamStatus("error");
      }
    })();

    return () => {
      cancelled = true;
      hlsRef.current?.destroy();
      hlsRef.current = null;
      pcRef.current?.close();
      pcRef.current = null;
      if (video) {
        video.srcObject = null;
        video.src = "";
      }
    };
  }, [hlsUrl, webrtcUrl]);

  return { videoRef, streamStatus };
}
