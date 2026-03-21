import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCameras, fetchCamera } from "@/services/api";
import { joinCameraGroup, leaveCameraGroup, onOmniEvent } from "@/services/signalr";
import { LiveStreamPlayer } from "@/components/LiveStreamPlayer";
import { parseBboxFromOmniEvent, type TrackOverlay } from "@/lib/parseOmniBbox";
import { useI18n } from "@/i18n/I18nProvider";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Terminal } from "lucide-react";
import type { OmniEvent } from "@/types/omni";

const MEDIA_BASE = import.meta.env.OMNI_MEDIA_BASE_URL ?? "http://localhost:8888";

export default function LivePreview() {
  const { t } = useI18n();
  const [cameraId, setCameraId] = useState<string | undefined>(undefined);
  const [showTracks, setShowTracks] = useState(true);
  const [tracks, setTracks] = useState<Map<string, TrackOverlay>>(() => new Map());

  const { data: cameras } = useQuery({ queryKey: ["cameras"], queryFn: fetchCameras });

  const { data: camera } = useQuery({
    queryKey: ["camera", cameraId],
    queryFn: () => fetchCamera(cameraId),
    enabled: !!cameraId,
  });

  const hlsUrl = useMemo(
    () => camera?.hlsUrl ?? (camera?.streamUrl ? `${MEDIA_BASE}/${camera.id}/index.m3u8` : undefined),
    [camera],
  );
  const webrtcUrl = useMemo(
    () => camera?.webrtcUrl ?? (camera?.streamUrl ? `${MEDIA_BASE}/${camera.id}/whep` : undefined),
    [camera],
  );

  const mergeEvent = useCallback(
    (event: OmniEvent) => {
      if (!cameraId || event.cameraId !== cameraId) return;
      const parsed = parseBboxFromOmniEvent(event);
      if (!parsed) return;
      setTracks((prev) => {
        const next = new Map(prev);
        next.set(parsed.id, { ...parsed, lastSeen: Date.now() });
        return next;
      });
    },
    [cameraId],
  );

  useEffect(() => {
    if (!cameraId) return;
    joinCameraGroup(cameraId);
    const unsub = onOmniEvent(mergeEvent);
    return () => {
      unsub();
      leaveCameraGroup(cameraId);
    };
  }, [cameraId, mergeEvent]);

  useEffect(() => {
    if (!cameraId) {
      setTracks(new Map());
      return;
    }
    const id = window.setInterval(() => {
      const cutoff = Date.now() - 800;
      setTracks((prev) => {
        const next = new Map<string, TrackOverlay>();
        for (const [k, v] of prev) {
          if (v.lastSeen >= cutoff) next.set(k, v);
        }
        return next;
      });
    }, 200);
    return () => clearInterval(id);
  }, [cameraId]);

  const trackList = useMemo(() => [...tracks.values()].sort((a, b) => b.lastSeen - a.lastSeen), [tracks]);
  const freshCount = trackList.filter((x) => Date.now() - x.lastSeen < 600).length;

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">{t("live.title")}</h1>
        <p className="text-muted-foreground text-sm mt-1">{t("live.subtitle")}</p>
      </div>

      <Alert className="border-amber-500/40 bg-amber-500/5">
        <Terminal className="h-4 w-4" />
        <AlertTitle>{t("live.troubleshootTitle")}</AlertTitle>
        <AlertDescription className="text-xs mt-2 space-y-2 [&_code]:block [&_code]:text-[11px] [&_code]:bg-muted [&_code]:px-2 [&_code]:py-1 [&_code]:rounded [&_code]:mt-1">
          <p>{t("live.troubleshootP1")}</p>
          <code>docker compose --profile media up -d omni-mediamtx</code>
          <p>{t("live.troubleshootP2")}</p>
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{t("live.selectCamera")}</CardTitle>
          <CardDescription>{t("live.tracksHint")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <div className="flex-1 space-y-2">
              <Label>{t("live.selectCamera")}</Label>
              <Select
                value={cameraId || undefined}
                onValueChange={(v) => setCameraId(v)}
              >
                <SelectTrigger>
                  <SelectValue placeholder={t("live.pickCamera")} />
                </SelectTrigger>
                <SelectContent>
                  {(cameras ?? []).map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      {c.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-3 pt-6 sm:pt-8">
              <Switch id="tracks" checked={showTracks} onCheckedChange={setShowTracks} />
              <Label htmlFor="tracks" className="cursor-pointer">
                {t("live.showTracks")}
              </Label>
            </div>
          </div>

          {!cameraId && (
            <p className="text-sm text-muted-foreground">{t("live.pickCamera")}</p>
          )}

          {cameraId && !hlsUrl && !webrtcUrl && (
            <p className="text-sm text-amber-600 dark:text-amber-400">{t("live.noStream")}</p>
          )}

          {cameraId && (
            <>
              <LiveStreamPlayer
                hlsUrl={hlsUrl}
                webrtcUrl={webrtcUrl}
                tracks={trackList}
                showOverlays={showTracks}
              />
              <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                <Badge variant="secondary">
                  {t("live.activeBoxes")}: {freshCount}
                </Badge>
                {freshCount === 0 && showTracks && (
                  <span>{t("live.waitingEvents")}</span>
                )}
              </div>
              <p className="text-xs text-muted-foreground">{t("live.joinSignalR")}</p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
