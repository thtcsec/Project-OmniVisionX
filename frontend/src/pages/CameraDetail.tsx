import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, useMemo } from "react";
import { fetchCamera, fetchDetections } from "@/services/api";
import { joinCameraGroup, leaveCameraGroup, onOmniEvent } from "@/services/signalr";
import { VideoPlayer } from "@/components/VideoPlayer";
import { EventFeed } from "@/components/EventFeed";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { OmniEvent, Detection } from "@/types/omni";

const MEDIA_BASE = import.meta.env.OMNI_MEDIA_BASE_URL ?? "http://localhost:8888";

export default function CameraDetail() {
  const { id } = useParams<{ id: string }>();
  const { data: camera, isLoading } = useQuery({
    queryKey: ["camera", id],
    queryFn: () => fetchCamera(id!),
    enabled: !!id,
  });

  // Live events for this camera
  const [liveEvents, setLiveEvents] = useState<OmniEvent[]>([]);
  useEffect(() => {
    if (!id) return;
    joinCameraGroup(id);
    const unsub = onOmniEvent((event) => {
      if (event.cameraId === id) {
        setLiveEvents((prev) => [event, ...prev].slice(0, 50));
      }
    });
    return () => {
      unsub();
      leaveCameraGroup(id);
    };
  }, [id]);

  // History filters
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ["detections", id, dateFrom, dateTo],
    queryFn: () => fetchDetections({ cameraId: id!, from: dateFrom || undefined, to: dateTo || undefined }),
    enabled: !!id,
  });

  // Build stream URLs from camera or fallback
  const hlsUrl = useMemo(() => camera?.hlsUrl ?? (camera?.streamUrl ? `${MEDIA_BASE}/${camera.id}/index.m3u8` : undefined), [camera]);
  const webrtcUrl = useMemo(() => camera?.webrtcUrl ?? (camera?.streamUrl ? `${MEDIA_BASE}/${camera.id}/whep` : undefined), [camera]);

  if (isLoading) {
    return <div className="space-y-4 max-w-5xl"><Skeleton className="h-8 w-48" /><Skeleton className="aspect-video w-full" /></div>;
  }

  if (!camera) {
    return <div className="text-muted-foreground py-12 text-center">Camera not found.</div>;
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-bold text-foreground">{camera.name}</h1>
        <Badge variant={camera.status === "online" ? "default" : "secondary"}>{camera.status}</Badge>
      </div>

      <VideoPlayer hlsUrl={hlsUrl} webrtcUrl={webrtcUrl} />

      <Tabs defaultValue="live">
        <TabsList>
          <TabsTrigger value="live">Live Events</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="live">
          <Card>
            <CardHeader><CardTitle className="text-base">Real-time Detections</CardTitle></CardHeader>
            <CardContent><EventFeed events={liveEvents} maxHeight="400px" /></CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Detection History</CardTitle>
              <div className="flex gap-3 mt-2">
                <Input type="datetime-local" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="w-auto" />
                <Input type="datetime-local" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="w-auto" />
              </div>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <Skeleton className="h-32 w-full" />
              ) : !history || history.length === 0 ? (
                <p className="text-sm text-muted-foreground py-8 text-center">No detections found for this period.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {history.map((d: Detection) => (
                      <TableRow key={d.id}>
                        <TableCell className="font-mono text-xs">{new Date(d.timestamp).toLocaleString()}</TableCell>
                        <TableCell><Badge variant="outline">{d.type}</Badge></TableCell>
                        <TableCell>{(d.confidence * 100).toFixed(1)}%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
