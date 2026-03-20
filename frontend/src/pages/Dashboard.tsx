import { useQuery } from "@tanstack/react-query";
import { useOutletContext } from "react-router-dom";
import { fetchDashboardStats } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EventFeed } from "@/components/EventFeed";
import { Skeleton } from "@/components/ui/skeleton";
import { Camera, Eye, ScanLine, AlertTriangle } from "lucide-react";
import type { OmniEvent } from "@/types/omni";

const statCards = [
  { key: "camerasOnline" as const, label: "Cameras Online", icon: Camera, format: (v: number, t?: number) => t != null ? `${v}/${t}` : `${v}` },
  { key: "detectionsToday" as const, label: "Detections Today", icon: Eye, format: (v: number) => v.toLocaleString() },
  { key: "platesDetected" as const, label: "Plates Detected", icon: ScanLine, format: (v: number) => v.toLocaleString() },
  { key: "activeAlerts" as const, label: "Active Alerts", icon: AlertTriangle, format: (v: number) => v.toLocaleString() },
];

export default function Dashboard() {
  const { events } = useOutletContext<{ events: OmniEvent[] }>();
  const { data: stats, isLoading, isError } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: fetchDashboardStats,
    refetchInterval: 30000,
    retry: 2,
  });

  return (
    <div className="space-y-6 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">Real-time traffic and vision analytics overview</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map(({ key, label, icon: Icon, format }) => (
          <Card key={key}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-8 w-20" />
              ) : isError || !stats ? (
                <span className="text-sm text-muted-foreground">—</span>
              ) : (
                <p className="text-2xl font-bold text-foreground">
                  {key === "camerasOnline" ? format(stats.camerasOnline, stats.camerasTotal) : format(stats[key])}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recent events */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent Events</CardTitle>
        </CardHeader>
        <CardContent>
          <EventFeed events={events.slice(0, 15)} maxHeight="360px" />
        </CardContent>
      </Card>
    </div>
  );
}
