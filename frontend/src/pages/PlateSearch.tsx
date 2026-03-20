import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchPlates } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Search, Download } from "lucide-react";

export default function PlateSearch() {
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");

  const { data: plates, isLoading, isError } = useQuery({
    queryKey: ["plates", search],
    queryFn: () => fetchPlates(search || undefined),
    enabled: true,
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
                  <TableHead>Plate</TableHead>
                  <TableHead>Camera</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Timestamp</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {plates.map((p) => (
                  <TableRow key={p.id}>
                    <TableCell className="font-mono font-semibold">{p.plateText}</TableCell>
                    <TableCell>{p.cameraName ?? p.cameraId}</TableCell>
                    <TableCell>{(p.confidence * 100).toFixed(1)}%</TableCell>
                    <TableCell className="font-mono text-xs">{new Date(p.timestamp).toLocaleString()}</TableCell>
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
