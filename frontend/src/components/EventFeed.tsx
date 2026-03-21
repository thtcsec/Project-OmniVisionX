import type { OmniEvent } from "@/types/omni";
import { parseOmniEventPayload } from "@/lib/parseOmniBbox";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Car, User, ScanLine, Eye } from "lucide-react";

const typeIcons: Record<string, React.ElementType> = {
  vehicle: Car,
  human: User,
  plate: ScanLine,
  detection: Eye,
};

const typeColors: Record<string, string> = {
  vehicle: "bg-primary/10 text-primary",
  human: "bg-accent text-accent-foreground",
  plate: "bg-destructive/10 text-destructive",
  detection: "bg-secondary text-secondary-foreground",
};

interface EventFeedProps {
  events: OmniEvent[];
  maxHeight?: string;
}

export function EventFeed({ events, maxHeight = "400px" }: EventFeedProps) {
  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground text-sm">
        No events yet — waiting for detections…
      </div>
    );
  }

  return (
    <ScrollArea style={{ maxHeight }}>
      <div className="space-y-2 pr-2">
        {events.map((event, i) => {
          const Icon = typeIcons[event.type] ?? Eye;
          const colorClass = typeColors[event.type] ?? typeColors.detection;
          const time = new Date(event.timestamp).toLocaleTimeString();
          const payload = parseOmniEventPayload(event.data);
          const summary = payload
            ? String(
                (payload.label as string | undefined) ??
                  (payload.plateText as string | undefined) ??
                  (payload.type as string | undefined) ??
                  event.type,
              )
            : event.type;

          return (
            <div key={`${event.timestamp}-${i}`} className="flex items-start gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors">
              <div className={`p-1.5 rounded-md ${colorClass}`}>
                <Icon className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium truncate">{summary}</span>
                  <Badge variant="outline" className="text-[10px] shrink-0">{event.type}</Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Camera {event.cameraId} · {time}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
