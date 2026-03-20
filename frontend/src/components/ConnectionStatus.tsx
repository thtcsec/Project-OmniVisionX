import type { ConnectionStatus as Status } from "@/hooks/useSignalR";
import { Wifi, WifiOff, Loader2 } from "lucide-react";

const config: Record<Status, { icon: React.ElementType; label: string; className: string }> = {
  connected: { icon: Wifi, label: "Live", className: "text-emerald-600 bg-emerald-50" },
  disconnected: { icon: WifiOff, label: "Offline", className: "text-destructive bg-destructive/10" },
  reconnecting: { icon: Loader2, label: "Reconnecting", className: "text-muted-foreground bg-muted" },
};

export function ConnectionStatus({ status }: { status: Status }) {
  const { icon: Icon, label, className } = config[status];
  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${className}`}>
      <Icon className={`h-3 w-3 ${status === "reconnecting" ? "animate-spin" : ""}`} />
      {label}
    </div>
  );
}
