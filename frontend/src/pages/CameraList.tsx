import { useQuery } from "@tanstack/react-query";
import { fetchCameras } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Link } from "react-router-dom";
import { Camera, Eye, ScanLine, User } from "lucide-react";

export default function CameraList() {
  const { data: cameras, isLoading, isError } = useQuery({
    queryKey: ["cameras"],
    queryFn: fetchCameras,
    refetchInterval: 15000,
  });

  return (
    <div className="space-y-6 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Cameras</h1>
        <p className="text-muted-foreground text-sm mt-1">Manage and monitor connected cameras</p>
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}><CardContent className="p-6"><Skeleton className="h-24 w-full" /></CardContent></Card>
          ))}
        </div>
      )}

      {isError && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            Failed to load cameras. Check that the API is running.
          </CardContent>
        </Card>
      )}

      {cameras && cameras.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No cameras configured yet.
          </CardContent>
        </Card>
      )}

      {cameras && cameras.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cameras.map((cam) => (
            <Link to={`/cameras/${cam.id}`} key={cam.id}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Camera className="h-4 w-4" />
                      {cam.name}
                    </CardTitle>
                    <Badge variant={cam.status === "online" ? "default" : "secondary"}>
                      {cam.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex gap-2 flex-wrap">
                    {cam.features.objectDetection && (
                      <Badge variant="outline" className="text-[10px] gap-1"><Eye className="h-3 w-3" />Objects</Badge>
                    )}
                    {cam.features.plateRecognition && (
                      <Badge variant="outline" className="text-[10px] gap-1"><ScanLine className="h-3 w-3" />Plates</Badge>
                    )}
                    {cam.features.faceDetection && (
                      <Badge variant="outline" className="text-[10px] gap-1"><User className="h-3 w-3" />Faces</Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
