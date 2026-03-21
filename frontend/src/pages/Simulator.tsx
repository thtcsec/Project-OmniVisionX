import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchSimulatorVideos,
  fetchSimulatorCameras,
  rescanSimulatorVideos,
  startSimulatorCamera,
  stopSimulatorCamera,
} from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Square, Film, RefreshCcw, Radio } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { useI18n } from "@/i18n/I18nProvider";
import type { SimulatorVideo } from "@/types/omni";

function simulatorCameraId(video: SimulatorVideo): string {
  const stem = (video.filename || video.name || "").replace(/\.[^.]+$/, "") || video.name;
  return `cam-${stem}`;
}

function resolveVideoPath(video: SimulatorVideo): string | undefined {
  if (video.path?.trim()) return video.path.trim();
  if (video.filename) return `/videos/${video.filename}`;
  return undefined;
}

export default function Simulator() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const { data: videos, isLoading: vLoading } = useQuery({ queryKey: ["sim-videos"], queryFn: fetchSimulatorVideos });
  const { data: cameras, isLoading: cLoading } = useQuery({
    queryKey: ["sim-cameras"],
    queryFn: fetchSimulatorCameras,
    refetchInterval: 5000,
  });

  const rescanMut = useMutation({
    mutationFn: () => rescanSimulatorVideos(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sim-videos"] });
      toast({ title: t("simulator.videosRescanned") });
    },
    onError: () => toast({ title: t("simulator.rescanFailed"), variant: "destructive" }),
  });

  const startMut = useMutation({
    mutationFn: ({ cameraId, videoPath }: { cameraId: string; videoPath: string }) =>
      startSimulatorCamera(cameraId, videoPath),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sim-cameras"] });
      toast({ title: t("simulator.streamStarted") });
    },
    onError: () => toast({ title: t("simulator.streamStartFailed"), variant: "destructive" }),
  });

  const stopMut = useMutation({
    mutationFn: stopSimulatorCamera,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sim-cameras"] });
      toast({ title: t("simulator.streamStopped") });
    },
    onError: () => toast({ title: t("simulator.stopFailed"), variant: "destructive" }),
  });

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">{t("simulator.title")}</h1>
        <p className="text-muted-foreground text-sm mt-1">{t("simulator.subtitle")}</p>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Film className="h-4 w-4" />
            {t("simulator.availableVideos")}
          </CardTitle>
          <Button size="sm" variant="outline" onClick={() => rescanMut.mutate()} disabled={rescanMut.isPending}>
            <RefreshCcw className="h-3 w-3 mr-1" />
            {t("simulator.rescan")}
          </Button>
        </CardHeader>
        <CardContent>
          {vLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : !videos || videos.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">{t("simulator.noVideos")}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t("simulator.name")}</TableHead>
                  <TableHead>{t("simulator.file")}</TableHead>
                  <TableHead>{t("simulator.streamId")}</TableHead>
                  <TableHead>{t("simulator.duration")}</TableHead>
                  <TableHead className="text-right">{t("simulator.actions")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {videos.map((v) => {
                  const camId = simulatorCameraId(v);
                  const fsPath = resolveVideoPath(v);
                  const running = cameras?.some((c) => c.id === camId && c.status === "running");
                  return (
                    <TableRow key={v.id}>
                      <TableCell className="font-medium">{v.name}</TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">{v.filename}</TableCell>
                      <TableCell className="font-mono text-xs">{camId}</TableCell>
                      <TableCell>{v.duration ? `${v.duration}s` : "—"}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant={running ? "secondary" : "default"}
                          disabled={
                            !fsPath ||
                            (startMut.isPending && startMut.variables?.cameraId === camId)
                          }
                          onClick={() => {
                            if (!fsPath) {
                              toast({ title: t("simulator.missingPath"), variant: "destructive" });
                              return;
                            }
                            startMut.mutate({ cameraId: camId, videoPath: fsPath });
                          }}
                        >
                          <Radio className="h-3 w-3 mr-1" />
                          {startMut.isPending && startMut.variables?.cameraId === camId
                            ? t("simulator.generating")
                            : t("simulator.generateStream")}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t("simulator.cameraStreams")}</CardTitle>
        </CardHeader>
        <CardContent>
          {cLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : !cameras || cameras.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">{t("simulator.noCameras")}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>{t("simulator.rtspUrl")}</TableHead>
                  <TableHead>{t("simulator.status")}</TableHead>
                  <TableHead className="text-right">{t("simulator.action")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {cameras.map((cam) => (
                  <TableRow key={cam.id}>
                    <TableCell className="font-mono text-xs">{cam.id}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground truncate max-w-[300px]">
                      {cam.rtspUrl}
                    </TableCell>
                    <TableCell>
                      <Badge variant={cam.status === "running" ? "default" : "secondary"}>{cam.status}</Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {cam.status === "running" ? (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => stopMut.mutate(cam.id)}
                          disabled={stopMut.isPending}
                        >
                          <Square className="h-3 w-3 mr-1" />
                          {t("simulator.stop")}
                        </Button>
                      ) : (
                        <span className="text-xs text-muted-foreground pr-2">
                          {t("simulator.generateStream")} ↑
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
