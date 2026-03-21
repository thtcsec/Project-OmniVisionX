import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { chatWithCamera, fetchAgoraStatus, fetchCameras, fetchCamera, fetchIntegrationsStatus, fetchLatestDetectionsSnapshot, speakText } from "@/services/api";
import {
  joinCameraGroup,
  leaveCameraGroup,
  onOmniEvent,
  onConnectionStatus,
  startConnection,
} from "@/services/signalr";
import { LiveStreamPlayer } from "@/components/LiveStreamPlayer";
import { AgoraPlayer } from "@/components/AgoraPlayer";
import { parseBboxFromOmniEvent, type TrackOverlay } from "@/lib/parseOmniBbox";
import { useI18n } from "@/i18n/I18nProvider";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { ChevronLeft, ChevronRight, CircleHelp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { OmniEvent } from "@/types/omni";

import { buildCameraHlsUrl, buildCameraWebRtcUrl } from "@/lib/mediaUrls";
import { OVERLAY_TRACK_TTL_MS } from "@/lib/overlayConstants";
const OMNI_STREAM_URL = (import.meta.env.OMNI_STREAM_URL as string | undefined)?.trim();


const CAMERA_NONE = "__none__";
const PAGE_SIZE = 10;

export default function LivePreview() {
  const { t } = useI18n();
  const [cameraId, setCameraId] = useState<string>(CAMERA_NONE);
  const [page, setPage] = useState(1);
  const [showTracks, setShowTracks] = useState(false);
  const [tracks, setTracks] = useState<Map<string, TrackOverlay>>(() => new Map());
  const [signalRStatus, setSignalRStatus] = useState<"connected" | "disconnected" | "reconnecting">("disconnected");
  const [chatText, setChatText] = useState("");
  const [chatLog, setChatLog] = useState<Array<{ role: "user" | "assistant"; content: string; meta?: string }>>([]);

  const { data: cameras } = useQuery({ queryKey: ["cameras"], queryFn: fetchCameras });

  const list = cameras ?? [];
  const totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const paginatedCameras = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return list.slice(start, start + PAGE_SIZE);
  }, [list, page]);

  const { data: camera } = useQuery({
    queryKey: ["camera", cameraId],
    queryFn: () => fetchCamera(cameraId),
    enabled: !!cameraId && cameraId !== CAMERA_NONE,
  });

  const mjpegUrl = OMNI_STREAM_URL && cameraId !== CAMERA_NONE ? OMNI_STREAM_URL : undefined;

  const hlsUrl = useMemo(
    () => camera?.hlsUrl ?? (camera?.streamUrl ? buildCameraHlsUrl(camera) : undefined),
    [camera],
  );
  const webrtcUrl = useMemo(
    () => camera?.webrtcUrl ?? (camera?.streamUrl ? buildCameraWebRtcUrl(camera) : undefined),
    [camera],
  );

  const mergeEvent = useCallback(
    (event: OmniEvent) => {
      const evCam = String(event.cameraId ?? "").trim();
      const selected = String(cameraId ?? "").trim();
      if (!selected || selected === CAMERA_NONE || evCam !== selected) return;
      const parsed = parseBboxFromOmniEvent(event);
      if (!parsed) return;
      const sourceTs = Date.parse(String(event.timestamp ?? ""));
      const sourceTsMs = Number.isFinite(sourceTs) ? sourceTs : Date.now();
      setTracks((prev) => {
        const next = new Map(prev);
        next.set(parsed.id, { ...parsed, lastSeen: Date.now(), sourceTs: sourceTsMs } as TrackOverlay);
        return next;
      });
    },
    [cameraId],
  );

  /** Direct omni-object snapshot — works even when Redis→API→SignalR path is down. */
  const snapshotQuery = useQuery({
    queryKey: ["latestDetectionsSnapshot", cameraId],
    queryFn: () => fetchLatestDetectionsSnapshot(cameraId),
    enabled: !!cameraId && cameraId !== CAMERA_NONE && showTracks && !mjpegUrl,
    refetchInterval: 500,
    retry: 1,
  });

  useEffect(() => {
    if (!showTracks || cameraId === CAMERA_NONE) return;
    const block = snapshotQuery.data?.items?.[cameraId];
    const dets = block?.detections;
    if (!dets?.length) return;
    const ts = Date.now();
    const blockSourceTs = typeof block?.timestamp === "number" ? block.timestamp * 1000 : ts;
    setTracks((prev) => {
      const next = new Map(prev);
      for (const d of dets) {
        if (!Array.isArray(d.bbox) || d.bbox.length !== 4) continue;
        const trackId = Number(d.track_id);
        if (!Number.isFinite(trackId) || trackId < 0) continue;
        const [x1, y1, x2, y2] = d.bbox;
        const id = String(trackId);
        next.set(id, {
          id,
          x1,
          y1,
          x2,
          y2,
          label: String(d.class_name ?? d.class ?? "detection"),
          confidence: Number(d.confidence ?? 0),
          lastSeen: ts,
          kind: "detection",
          sourceTs: blockSourceTs,
        } as TrackOverlay);
      }
      return next;
    });
  }, [snapshotQuery.data, cameraId, showTracks]);

  const integrationsQuery = useQuery({
    queryKey: ["integrations-status"],
    queryFn: fetchIntegrationsStatus,
    staleTime: 60_000,
    retry: false,
  });

  const agoraStatusQuery = useQuery({
    queryKey: ["agora-status"],
    queryFn: fetchAgoraStatus,
    staleTime: 60_000,
    retry: false,
  });

  const llmConfigured = !!(integrationsQuery.data?.openai.configured || integrationsQuery.data?.qwen.configured);
  const ttsConfigured = !!integrationsQuery.data?.elevenlabs.configured;
  const agoraConfigured = !!agoraStatusQuery.data?.configured;

  const chatMut = useMutation({
    mutationFn: (payload: { message: string; cameraId?: string }) => chatWithCamera(payload),
    onSuccess: (data) => {
      setChatLog((prev) => [...prev, { role: "assistant", content: data.reply, meta: `${data.provider}:${data.model}` }]);
    },
    onError: (err) => {
      setChatLog((prev) => [...prev, { role: "assistant", content: err instanceof Error ? err.message : "Chat failed", meta: "error" }]);
    },
  });

  useEffect(() => {
    const unsub = onConnectionStatus(setSignalRStatus);
    void startConnection();
    return () => unsub();
  }, []);

  useEffect(() => {
    if (!cameraId || cameraId === CAMERA_NONE) return;
    const unsub = onOmniEvent(mergeEvent);
    void joinCameraGroup(cameraId);
    return () => {
      unsub();
      void leaveCameraGroup(cameraId);
    };
  }, [cameraId, mergeEvent]);

  useEffect(() => {
    // Clear tracks IMMEDIATELY when camera changes (prevents bbox bleed from old camera)
    setTracks(new Map());
    if (!cameraId || cameraId === CAMERA_NONE) {
      setChatLog([]);
      return;
    }
    // Periodic stale-track eviction
    const id = window.setInterval(() => {
      const cutoff = Date.now() - OVERLAY_TRACK_TTL_MS;
      setTracks((prev) => {
        const next = new Map<string, TrackOverlay>();
        for (const [k, v] of prev) {
          if (v.lastSeen >= cutoff) next.set(k, v);
        }
        return next;
      });
    }, 1000);
    return () => clearInterval(id);
  }, [cameraId]);

  const trackList = useMemo(() => [...tracks.values()].sort((a, b) => b.lastSeen - a.lastSeen), [tracks]);
  const freshCount = trackList.filter((x) => Date.now() - x.lastSeen < OVERLAY_TRACK_TTL_MS).length;
  const canChat = llmConfigured && cameraId !== CAMERA_NONE && !chatMut.isPending;
  const ttsMut = useMutation({
    mutationFn: (payload: { text: string }) => speakText(payload),
  });

  const sendChat = useCallback(() => {
    const msg = chatText.trim();
    if (!msg) return;
    if (!llmConfigured) return;
    if (cameraId === CAMERA_NONE) return;
    setChatText("");
    setChatLog((prev) => [...prev, { role: "user", content: msg }]);
    chatMut.mutate({ message: msg, cameraId });
  }, [chatText, llmConfigured, cameraId, chatMut]);

  return (
    <div className="mx-auto flex max-w-[1600px] flex-col gap-4">
      {/* Title + optional help */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-foreground">{t("live.title")}</h1>
          <p className="text-muted-foreground mt-1 text-sm">{t("live.subtitle")}</p>
        </div>
        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="shrink-0"
              aria-label={t("live.streamHelpLabel")}
            >
              <CircleHelp className="h-4 w-4" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-[min(100vw-2rem,24rem)] space-y-2 text-xs">
            <p className="font-semibold text-foreground">{t("live.troubleshootTitle")}</p>
            <p className="text-muted-foreground">{t("live.troubleshootP1")}</p>
            <code className="block rounded bg-muted px-2 py-1 text-[11px]">docker compose up -d omni-mediamtx</code>
            <p className="text-muted-foreground">{t("live.troubleshootP2")}</p>
          </PopoverContent>
        </Popover>
      </div>

      {/* Cameras on top — horizontal strip */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-base">{t("live.cameraListTitle")}</CardTitle>
              <CardDescription className="text-xs">{t("live.cameraListHint")}</CardDescription>
            </div>
            {totalPages > 1 && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 shrink-0"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  aria-label={t("live.prevPage")}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="tabular-nums">
                  {t("live.pageIndicator")
                    .replace("{page}", String(page))
                    .replace("{total}", String(totalPages))}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 shrink-0"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  aria-label={t("live.nextPage")}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {list.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t("live.noCamerasInList")}</p>
          ) : (
            <div className="flex max-h-[min(40vh,280px)] flex-wrap gap-2 overflow-y-auto pr-1">
              <Button
                type="button"
                variant={cameraId === CAMERA_NONE ? "secondary" : "outline"}
                size="sm"
                className="h-auto shrink-0 max-w-[min(100%,20rem)] justify-start whitespace-normal py-2 text-left font-normal"
                onClick={() => setCameraId(CAMERA_NONE)}
              >
                {t("live.pickCamera")}
              </Button>
              {paginatedCameras.map((c) => (
                <Button
                  key={c.id}
                  type="button"
                  variant={cameraId === c.id ? "default" : "outline"}
                  size="sm"
                  className={cn(
                    "h-auto shrink-0 max-w-[min(100%,20rem)] justify-start whitespace-normal py-2 text-left font-normal",
                    cameraId === c.id && "ring-2 ring-primary/30",
                  )}
                  onClick={() => setCameraId(c.id)}
                >
                  <span className="line-clamp-2">{c.name}</span>
                </Button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preview */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-base">{t("live.previewTitle")}</CardTitle>
              <CardDescription className="mt-1">{t("live.tracksHint")}</CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <Switch id="tracks" checked={showTracks} onCheckedChange={setShowTracks} />
                <Label htmlFor="tracks" className="cursor-pointer whitespace-nowrap">
                  {t("live.showTracks")}
                </Label>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {cameraId === CAMERA_NONE && (
            <p className="text-muted-foreground text-sm">{t("live.pickCamera")}</p>
          )}

          {cameraId !== CAMERA_NONE && !mjpegUrl && !hlsUrl && !webrtcUrl && (
            <p className="text-sm text-amber-600 dark:text-amber-400">{t("live.noStream")}</p>
          )}

          {cameraId !== CAMERA_NONE && (
            <>
              {mjpegUrl && (
                <p className="text-sm text-emerald-700 dark:text-emerald-400">Server-side overlay stream is active (OMNI_STREAM_URL).</p>
              )}
              {!showTracks && (
                <p className="text-sm text-amber-700 dark:text-amber-400">{t("live.overlayOffHint")}</p>
              )}
              {mjpegUrl ? (
                <div className="relative overflow-hidden rounded-lg bg-black">
                  <img src={mjpegUrl} alt="omni-stream" className="aspect-video h-full w-full object-contain" />
                </div>
              ) : agoraConfigured && cameraId !== CAMERA_NONE ? (
                <AgoraPlayer channelName={cameraId} />
              ) : (
                <LiveStreamPlayer
                  hlsUrl={hlsUrl}
                  webrtcUrl={webrtcUrl}
                  tracks={trackList}
                  showOverlays={showTracks}
                />
              )}
              <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-sm">
                <Badge
                  variant={
                    signalRStatus === "connected"
                      ? "default"
                      : signalRStatus === "reconnecting"
                        ? "secondary"
                        : "destructive"
                  }
                >
                  {signalRStatus === "connected"
                    ? t("live.signalRConnected")
                    : signalRStatus === "reconnecting"
                      ? t("live.signalRReconnecting")
                      : t("live.signalRDisconnected")}
                </Badge>
                {agoraConfigured && (
                  <Badge variant="outline" className="border-sky-500/50 text-sky-700 dark:text-sky-300">
                    Agora RTC
                  </Badge>
                )}
                <Badge variant="secondary">
                  {t("live.activeBoxes")}: {freshCount}
                </Badge>
                {(snapshotQuery.isError || snapshotQuery.data?.error) && (
                  <Badge variant="outline" className="border-amber-500/50 text-amber-800 dark:text-amber-300">
                    {t("live.snapshotError")}
                  </Badge>
                )}
                {snapshotQuery.isSuccess &&
                  !snapshotQuery.data?.error &&
                  (snapshotQuery.data?.items?.[cameraId]?.detections?.length ?? 0) > 0 && (
                    <Badge variant="outline">{t("live.snapshotOk")}</Badge>
                  )}
                {freshCount === 0 && showTracks && <span>{t("live.waitingEvents")}</span>}
              </div>
              <p className="text-muted-foreground text-xs">{t("live.pipelineHint")}</p>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("live.chatTitle")}</CardTitle>
          <CardDescription className="mt-1">{t("live.chatHint")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {!llmConfigured && (
            <p className="text-sm text-muted-foreground">{t("live.chatNeedConfig")}</p>
          )}
          <ScrollArea className="h-[220px] rounded-md border p-3">
            <div className="space-y-2">
              {chatLog.length === 0 && (
                <p className="text-sm text-muted-foreground">{t("live.chatEmpty")}</p>
              )}
              {chatLog.map((m, idx) => (
                <div key={idx} className={cn("text-sm", m.role === "user" ? "text-foreground" : "text-muted-foreground")}>
                  <span className="font-medium">{m.role === "user" ? t("live.chatYou") : t("live.chatAssistant")}: </span>
                  <span>{m.content}</span>
                  {m.meta && <span className="ml-2 text-xs text-muted-foreground">({m.meta})</span>}
                  {ttsConfigured && m.role === "assistant" && (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      className="ml-2 h-6 px-2"
                      disabled={ttsMut.isPending}
                      onClick={async () => {
                        const blob = await ttsMut.mutateAsync({ text: m.content });
                        const url = URL.createObjectURL(blob);
                        const audio = new Audio(url);
                        audio.onended = () => URL.revokeObjectURL(url);
                        void audio.play();
                      }}
                    >
                      {t("live.chatSpeak")}
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>
          <div className="flex items-center gap-2">
            <Input
              placeholder={t("live.chatPlaceholder")}
              value={chatText}
              onChange={(e) => setChatText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendChat();
              }}
              disabled={!canChat}
            />
            <Button type="button" onClick={sendChat} disabled={!canChat}>
              {t("live.chatSend")}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
