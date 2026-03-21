import { useParams, useNavigate, Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, useMemo } from "react";
import { deleteCamera, fetchCamera, fetchDetections, updateCamera } from "@/services/api";
import { joinCameraGroup, leaveCameraGroup, onOmniEvent } from "@/services/signalr";
import { CameraFormDialog, type CameraFormValues } from "@/components/CameraFormDialog";
import { VideoPlayer } from "@/components/VideoPlayer";
import { EventFeed } from "@/components/EventFeed";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { ArrowLeft, Pencil, Trash2 } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import type { OmniEvent, Detection } from "@/types/omni";

import { buildDefaultHlsUrl, buildDefaultWebRtcUrl } from "@/lib/mediaUrls";

export default function CameraDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

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
  const hlsUrl = useMemo(
    () => camera?.hlsUrl ?? (camera?.streamUrl ? buildDefaultHlsUrl(camera.id) : undefined),
    [camera],
  );
  const webrtcUrl = useMemo(
    () => camera?.webrtcUrl ?? (camera?.streamUrl ? buildDefaultWebRtcUrl(camera.id) : undefined),
    [camera],
  );

  const updateMut = useMutation({
    mutationFn: (v: CameraFormValues) => updateCamera(id!, v),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["camera", id] });
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
      setEditOpen(false);
      toast({ title: "Camera updated" });
    },
    onError: (e: Error) => toast({ title: "Update failed", description: e.message, variant: "destructive" }),
  });

  const deleteMut = useMutation({
    mutationFn: () => deleteCamera(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
      queryClient.removeQueries({ queryKey: ["camera", id] });
      navigate("/cameras");
      toast({ title: "Camera deleted" });
    },
    onError: (e: Error) => toast({ title: "Delete failed", description: e.message, variant: "destructive" }),
  });

  if (isLoading) {
    return <div className="space-y-4 max-w-5xl"><Skeleton className="h-8 w-48" /><Skeleton className="aspect-video w-full" /></div>;
  }

  if (!camera) {
    return <div className="text-muted-foreground py-12 text-center">Camera not found.</div>;
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 flex-wrap">
          <Button variant="ghost" size="icon" asChild className="shrink-0">
            <Link to="/cameras" aria-label="Back to cameras">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <h1 className="text-2xl font-bold text-foreground">{camera.name}</h1>
          <Badge variant={camera.status === "online" ? "default" : "secondary"}>{camera.status}</Badge>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setEditOpen(true)}>
            <Pencil className="h-4 w-4 mr-2" />
            Edit
          </Button>
          <Button variant="outline" size="sm" className="text-destructive" onClick={() => setDeleteOpen(true)}>
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </Button>
        </div>
      </div>

      <CameraFormDialog
        open={editOpen}
        onOpenChange={setEditOpen}
        mode="edit"
        camera={camera}
        onSubmit={(v) => updateMut.mutateAsync(v)}
        isSubmitting={updateMut.isPending}
      />

      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this camera?</AlertDialogTitle>
            <AlertDialogDescription>
              Permanently remove “{camera.name}”. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteMut.mutate()}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

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
