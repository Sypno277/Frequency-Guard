import { useEffect, useState } from "react";
import { Server, Cpu, Database, Box, CheckCircle2, XCircle } from "lucide-react";
import { getModelInfo, type ModelInfo } from "@/lib/api/client";

interface ServerInfoBannerProps {
  refreshKey?: number;
}

/**
 * Backend / model status banner (Masterplan §5).
 *
 * Surfaces live model metadata (GET /api/v1/model) plus service reachability.
 * When the backend is offline, the banner flips to a clear diagnostic state
 * instead of hiding the problem.
 */
const ServerInfoBanner = ({ refreshKey = 0 }: ServerInfoBannerProps) => {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const info = await getModelInfo();
        if (!cancelled) {
          setModel(info);
          setReachable(true);
        }
      } catch {
        if (!cancelled) setReachable(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const status = reachable === null ? "checking" : reachable ? "online" : "offline";

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-4">
          <div
            className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              status === "online"
                ? "bg-success/15 text-success"
                : status === "offline"
                ? "bg-destructive/15 text-destructive"
                : "bg-muted text-muted-foreground"
            }`}
          >
            <Server className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-sm">Backend API</h3>
              <span
                className={`inline-flex items-center gap-1 text-[11px] font-medium rounded-full px-2 py-0.5 ${
                  status === "online"
                    ? "bg-success/15 text-success"
                    : status === "offline"
                    ? "bg-destructive/15 text-destructive"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {status === "online" && <CheckCircle2 className="h-3 w-3" />}
                {status === "offline" && <XCircle className="h-3 w-3" />}
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              FastAPI · frequency-domain inference · SQLite history
            </p>
          </div>
        </div>

        {model && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="flex items-center gap-2 rounded-lg bg-muted/30 px-3 py-2">
              <Box className="h-4 w-4 text-primary" />
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Version</div>
                <div className="font-mono text-sm font-medium">v{model.version}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg bg-muted/30 px-3 py-2">
              <Database className="h-4 w-4 text-primary" />
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Samples</div>
                <div className="font-mono text-sm font-medium">{model.n_training_samples ?? "—"}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg bg-muted/30 px-3 py-2">
              <Cpu className="h-4 w-4 text-primary" />
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Features</div>
                <div className="font-mono text-sm font-medium">{model.n_features ?? "—"}</div>
              </div>
            </div>
            <div className="flex items-center gap-2 rounded-lg bg-muted/30 px-3 py-2">
              <Database className="h-4 w-4 text-primary" />
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Mode</div>
                <div className="font-mono text-sm font-medium truncate max-w-[90px]">
                  {model.training_mode ?? "—"}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ServerInfoBanner;
