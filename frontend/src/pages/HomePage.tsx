import { useCallback, useState } from "react";
import { useOmniHub } from "@/hooks/useOmniHub";
import type { OmniEventPayload } from "@/services/omniHub";
import { getApiBaseUrl } from "@/services/env";

export function HomePage() {
  const [events, setEvents] = useState<OmniEventPayload[]>([]);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  const onEvent = useCallback((ev: OmniEventPayload) => {
    setEvents((prev) => [ev, ...prev].slice(0, 50));
  }, []);

  const { status, lastError, connect, disconnect } = useOmniHub(onEvent);

  const pingApi = useCallback(async () => {
    const base = getApiBaseUrl();
    const url = base ? `${base}/health` : "/health";
    try {
      const r = await fetch(url);
      setApiOk(r.ok);
    } catch {
      setApiOk(false);
    }
  }, []);

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold text-white">Control shell</h1>
        <p className="text-omni-muted text-sm max-w-prose">
          Minimal UI scaffold: check API health, connect to SignalR{" "}
          <code className="text-zinc-400">OmniHub</code>, and stream{" "}
          <code className="text-zinc-400">OmniEvent</code> payloads. Dev server proxies{" "}
          <code className="text-zinc-400">/api</code>, <code className="text-zinc-400">/hubs</code>
          , and <code className="text-zinc-400">/health</code> to{" "}
          <code className="text-zinc-400">OMNI_DEV_API_PROXY_TARGET</code> (default{" "}
          <code className="text-zinc-400">localhost:8080</code>).
        </p>
      </section>

      <section className="rounded-lg border border-omni-border bg-omni-surface p-4 space-y-3">
        <h2 className="text-sm font-medium text-zinc-300">API</h2>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void pingApi()}
            className="rounded-md bg-omni-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500"
          >
            GET /health
          </button>
          {apiOk === true && (
            <span className="text-sm text-emerald-400">OK</span>
          )}
          {apiOk === false && (
            <span className="text-sm text-red-400">Unreachable</span>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-omni-border bg-omni-surface p-4 space-y-3">
        <h2 className="text-sm font-medium text-zinc-300">SignalR</h2>
        <p className="text-xs text-omni-muted">
          Status: <span className="text-zinc-400">{status}</span>
          {lastError && (
            <span className="block text-red-400 mt-1">{lastError}</span>
          )}
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => void connect()}
            disabled={status === "connecting" || status === "live"}
            className="rounded-md border border-omni-border px-3 py-1.5 text-sm hover:bg-white/5 disabled:opacity-40"
          >
            Connect + SubscribeAll
          </button>
          <button
            type="button"
            onClick={() => void disconnect()}
            disabled={status === "idle"}
            className="rounded-md border border-omni-border px-3 py-1.5 text-sm hover:bg-white/5 disabled:opacity-40"
          >
            Disconnect
          </button>
        </div>
      </section>

      <section className="rounded-lg border border-omni-border bg-omni-surface p-4 space-y-2">
        <h2 className="text-sm font-medium text-zinc-300">Recent OmniEvent</h2>
        {events.length === 0 ? (
          <p className="text-xs text-omni-muted">No events yet.</p>
        ) : (
          <ul className="space-y-2 max-h-80 overflow-auto text-xs font-mono">
            {events.map((ev, i) => (
              <li
                key={`${ev.timestamp}-${i}`}
                className="border-b border-omni-border/60 pb-2 text-zinc-400"
              >
                <span className="text-omni-accent">{ev.type}</span>{" "}
                <span className="text-zinc-500">cam {ev.cameraId}</span>
                <pre className="mt-1 whitespace-pre-wrap break-all text-[11px]">
                  {ev.data}
                </pre>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
