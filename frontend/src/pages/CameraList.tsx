import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  createCamera,
  deleteCamera,
  fetchCameras,
  updateCamera,
} from "@/services/api";
import { onCamerasChanged } from "@/services/signalr";
import { CameraFormDialog } from "@/components/CameraFormDialog";
import type { CameraFormValues } from "@/components/CameraFormDialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
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
import { Camera, Eye, Pencil, Plus, ScanLine, Trash2, User } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import type { Camera as CameraType } from "@/types/omni";

export default function CameraList() {
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<CameraType | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CameraType | null>(null);

  const { data: cameras, isLoading, isError } = useQuery({
    queryKey: ["cameras"],
    queryFn: fetchCameras,
    refetchInterval: 15000,
  });

  useEffect(() => {
    return onCamerasChanged(() => {
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
    });
  }, [queryClient]);

  const createMut = useMutation({
    mutationFn: (v: CameraFormValues) => createCamera(v),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      setCreateOpen(false);
      toast({ title: "Camera created" });
    },
    onError: (e: Error) => toast({ title: "Create failed", description: e.message, variant: "destructive" }),
  });

  const updateMut = useMutation({
    mutationFn: ({ id, v }: { id: string; v: CameraFormValues }) => updateCamera(id, v),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
      queryClient.invalidateQueries({ queryKey: ["camera", id] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      setEditTarget(null);
      toast({ title: "Camera updated" });
    },
    onError: (e: Error) => toast({ title: "Update failed", description: e.message, variant: "destructive" }),
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteCamera(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ["cameras"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      queryClient.removeQueries({ queryKey: ["camera", id] });
      setDeleteTarget(null);
      toast({ title: "Camera deleted" });
    },
    onError: (e: Error) => toast({ title: "Delete failed", description: e.message, variant: "destructive" }),
  });

  return (
    <div className="space-y-6 max-w-7xl">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Cameras</h1>
          <p className="text-muted-foreground text-sm mt-1">
            CRUD cameras (hackathon — no login). Production should add JWT / API keys.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)} className="shrink-0">
          <Plus className="h-4 w-4 mr-2" />
          Add camera
        </Button>
      </div>

      <CameraFormDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        mode="create"
        onSubmit={(v) => createMut.mutateAsync(v)}
        isSubmitting={createMut.isPending}
      />
      <CameraFormDialog
        open={!!editTarget}
        onOpenChange={(o) => !o && setEditTarget(null)}
        mode="edit"
        camera={editTarget ?? undefined}
        onSubmit={(v) => editTarget && updateMut.mutateAsync({ id: editTarget.id, v })}
        isSubmitting={updateMut.isPending}
      />

      <AlertDialog open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete camera?</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget ? `Remove “${deleteTarget.name}” from the system. This cannot be undone.` : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteTarget && deleteMut.mutate(deleteTarget.id)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <Skeleton className="h-24 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {isError && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            Failed to load cameras. Check that the API is running (same origin / proxy).
          </CardContent>
        </Card>
      )}

      {cameras && cameras.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No cameras yet. Click <strong>Add camera</strong> to create one.
          </CardContent>
        </Card>
      )}

      {cameras && cameras.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cameras.map((cam) => (
            <Card key={cam.id} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <Link to={`/cameras/${cam.id}`} className="flex-1 min-w-0 group">
                    <CardTitle className="text-base flex items-center gap-2 group-hover:underline">
                      <Camera className="h-4 w-4 shrink-0" />
                      <span className="truncate">{cam.name}</span>
                    </CardTitle>
                  </Link>
                  <div className="flex items-center gap-1 shrink-0">
                    <Badge variant={cam.status === "online" ? "default" : "secondary"}>{cam.status}</Badge>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      aria-label="Edit"
                      onClick={(e) => {
                        e.preventDefault();
                        setEditTarget(cam);
                      }}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive"
                      aria-label="Delete"
                      onClick={(e) => {
                        e.preventDefault();
                        setDeleteTarget(cam);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <Link to={`/cameras/${cam.id}`} className="block">
                <CardContent className="pt-0">
                  <div className="flex gap-2 flex-wrap">
                    {cam.features.objectDetection && (
                      <Badge variant="outline" className="text-[10px] gap-1">
                        <Eye className="h-3 w-3" />
                        Objects
                      </Badge>
                    )}
                    {cam.features.plateRecognition && (
                      <Badge variant="outline" className="text-[10px] gap-1">
                        <ScanLine className="h-3 w-3" />
                        Plates
                      </Badge>
                    )}
                    {cam.features.faceDetection && (
                      <Badge variant="outline" className="text-[10px] gap-1">
                        <User className="h-3 w-3" />
                        Faces
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Link>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
