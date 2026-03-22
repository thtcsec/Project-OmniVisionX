import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchPlates } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { PlateResult } from "@/types/omni";
import { Search, Download } from "lucide-react";

export default function PlateSearch() {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<PlateResult | null>(null);

  const { data: plates, isLoading, isError, dataUpdatedAt, isFetching } = useQuery({
    queryKey: ["plates", search],
    queryFn: () => fetchPlates(search || undefined),
    enabled: true,
    /** DB-backed list — not SignalR; poll so new LPR rows from omni-vehicle appear without manual refresh. */
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
    staleTime: 0,
  });

  const handleSearch = () => setSearch(query);

  const exportCSV = () => {
    if (!plates || plates.length === 0) return;
    const header = "Plate,Camera,Confidence,Timestamp\n";
    const rows = plates.map((p) => `"${p.plateText}","${p.cameraName ?? p.cameraId}",${(p.confidence * 100).toFixed(1)}%,"${new Date(p.timestamp).toLocaleString()}"`).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `plates-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Plate Search</h1>
        <p className="text-muted-foreground text-sm mt-1">Search license plate detections</p>
        <p className="text-muted-foreground text-xs mt-2 max-w-xl">
          Data comes from Postgres (rows inserted by omni-vehicle LPR). This page polls the API about every 10s — it is
          not SignalR. No new rows usually means the LPR worker is down, wrong DB connection, or duplicate suppression
          (same plate + camera within the dedup window).
        </p>
      </div>

      <div className="flex gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search plate number…"
            className="pl-9"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
        </div>
        <Button onClick={handleSearch}>Search</Button>
        {plates && plates.length > 0 && (
          <Button variant="outline" onClick={exportCSV}>
            <Download className="h-4 w-4 mr-1" />CSV
          </Button>
        )}
        {dataUpdatedAt > 0 && (
          <span className="text-[11px] text-muted-foreground self-center tabular-nums">
            {isFetching ? "Updating…" : `Last fetch ${new Date(dataUpdatedAt).toLocaleTimeString()}`}
          </span>
        )}
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">Results</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : isError ? (
            <p className="text-sm text-muted-foreground py-8 text-center">Failed to load plates. Check the API.</p>
          ) : !plates || plates.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">No plates found.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Images</TableHead>
                  <TableHead>Plate</TableHead>
                  <TableHead>Camera</TableHead>
                  <TableHead>Vehicle</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Timestamp</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {plates.map((p) => (
                  <TableRow key={p.id} className="cursor-pointer" onClick={() => setSelected(p)}>
                    <TableCell className="py-2">
                      {p.plateImageUrl ? (
                        <img
                          src={p.plateImageUrl}
                          alt={p.plateText}
                          className="h-10 w-24 object-cover rounded border bg-muted"
                          loading="lazy"
                        />
                      ) : (
                        <div className="h-10 w-24 rounded border bg-muted" />
                      )}
                    </TableCell>
                    <TableCell className="font-mono font-semibold">{p.plateText}</TableCell>
                    <TableCell>{p.cameraName ?? p.cameraId}</TableCell>
                    <TableCell className="text-xs">{p.vehicleType ?? "unknown"}</TableCell>
                    <TableCell>{(p.confidence * 100).toFixed(1)}%</TableCell>
                    <TableCell className="font-mono text-xs">{new Date(p.timestamp).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle className="font-mono">{selected?.plateText}</DialogTitle>
          </DialogHeader>
          {selected && (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">Plate crop</p>
                {selected.plateImageUrl ? (
                  <img
                    src={selected.plateImageUrl}
                    alt={`${selected.plateText} plate`}
                    className="w-full rounded border bg-muted"
                    loading="lazy"
                  />
                ) : (
                  <div className="h-40 rounded border bg-muted" />
                )}
              </div>
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">Full frame</p>
                {selected.frameImageUrl ? (
                  <img
                    src={selected.frameImageUrl}
                    alt={`${selected.plateText} frame`}
                    className="w-full rounded border bg-muted"
                    loading="lazy"
                  />
                ) : (
                  <div className="h-40 rounded border bg-muted" />
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
