import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { useSignalR } from "@/hooks/useSignalR";
import { Outlet } from "react-router-dom";
import { useEffect, useRef } from "react";
import { toast } from "@/hooks/use-toast";

export function AppLayout() {
  const { status, events } = useSignalR();
  const hasConnected = useRef(false);

  useEffect(() => {
    if (status === "connected") {
      hasConnected.current = true;
    }
  }, [status]);

  useEffect(() => {
    if (status === "disconnected" && hasConnected.current) {
      toast({ title: "Connection lost", description: "Real-time updates are unavailable. Reconnecting…", variant: "destructive" });
    }
  }, [status]);

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <AppSidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-14 flex items-center justify-between border-b px-4 bg-background">
            <div className="flex items-center gap-3">
              <SidebarTrigger />
              <h2 className="text-lg font-semibold text-foreground hidden sm:block">OmniVision</h2>
            </div>
            <ConnectionStatus status={status} />
          </header>
          <main className="flex-1 p-6 overflow-auto">
            <Outlet context={{ events, connectionStatus: status }} />
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}
