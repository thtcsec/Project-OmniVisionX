import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchSimulatorVideos, fetchSimulatorCameras, startSimulatorCamera, stopSimulatorCamera } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Play, Square, Film } from "lucide-react";
import { toast } from "@/hooks/use-toast";

export default function Simulator() {
  const qc = useQueryClient();
  const { data: videos, isLoading: vLoading } = useQuery({ queryKey: ["sim-videos"], queryFn: fetchSimulatorVideos });
  const { data: cameras, isLoading: cLoading } = useQuery({ queryKey: ["sim-cameras"], queryFn: fetchSimulatorCameras, refetchInterval: 5000 });

  const startMut = useMutation({
    mutationFn: startSimulatorCamera,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["sim-cameras"] }); toast({ title: "Stream started" }); },
    onError: () => toast({ title: "Failed to start stream", variant: "destructive" }),
  });
  const stopMut = useMutation({
    mutationFn: stopSimulatorCamera,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["sim-cameras"] }); toast({ title: "Stream stopped" }); },
    onError: () => toast({ title: "Failed to stop stream", variant: "destructive" }),
  });

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Simulator</h1>
        <p className="text-muted-foreground text-sm mt-1">Manage mock camera streams for development</p>
      </div>

      {/* Videos */}
      <Card>
        <CardHeader><CardTitle className="text-base flex items-center gap-2"><Film className="h-4 w-4" />Available Videos</CardTitle></CardHeader>
        <CardContent>
          {vLoading ? <Skeleton className="h-24 w-full" /> : !videos || videos.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">No videos found on the simulator.</p>
          ) : (
            <Table>
              <TableHeader><TableRow><TableHead>Name</TableHead><TableHead>File</TableHead><TableHead>Duration</TableHead></TableRow></TableHeader>
              <TableBody>
                {videos.map((v) => (
                  <TableRow key={v.id}>
                    <TableCell className="font-medium">{v.name}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">{v.filename}</TableCell>
                    <TableCell>{v.duration ? `${v.duration}s` : "—"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Cameras / Streams */}
      <Card>
        <CardHeader><CardTitle className="text-base">Camera Streams</CardTitle></CardHeader>
        <CardContent>
          {cLoading ? <Skeleton className="h-24 w-full" /> : !cameras || cameras.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">No simulator cameras configured.</p>
          ) : (
            <Table>
              <TableHeader><TableRow><TableHead>ID</TableHead><TableHead>RTSP URL</TableHead><TableHead>Status</TableHead><TableHead className="text-right">Action</TableHead></TableRow></TableHeader>
              <TableBody>
                {cameras.map((cam) => (
                  <TableRow key={cam.id}>
                    <TableCell className="font-mono text-xs">{cam.id}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground truncate max-w-[300px]">{cam.rtspUrl}</TableCell>
                    <TableCell><Badge variant={cam.status === "running" ? "default" : "secondary"}>{cam.status}</Badge></TableCell>
                    <TableCell className="text-right">
                      {cam.status === "running" ? (
                        <Button size="sm" variant="outline" onClick={() => stopMut.mutate(cam.id)} disabled={stopMut.isPending}>
                          <Square className="h-3 w-3 mr-1" />Stop
                        </Button>
                      ) : (
                        <Button size="sm" onClick={() => startMut.mutate(cam.id)} disabled={startMut.isPending}>
                          <Play className="h-3 w-3 mr-1" />Start
                        </Button>
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
