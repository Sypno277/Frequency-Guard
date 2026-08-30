import { useEffect, useState } from 'react';
import {
  getModelInfo,
  getPerformance,
  type ModelInfo,
  type PerformanceReport
} from '@/lib/api/client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart3, TrendingUp, Target, Activity, AlertTriangle } from 'lucide-react';

/**
 * Training performance dashboard.
 *
 * All metrics come from the backend's persisted evaluation report
 * (GET /api/v1/model/performance) — no hardcoded or simulated numbers.
 * When no evaluation has been run the panels render explicit empty states.
 */
export default function TrainingDashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [perf, setPerf] = useState<PerformanceReport | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getModelInfo().catch(() => null), getPerformance().catch(() => null)]).then(
      ([m, p]) => {
        if (cancelled) return;
        setModel(m);
        setPerf(p);
        setUnavailable(m === null && p === null);
      }
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const pct = (v: number | null | undefined) =>
    v != null ? `${(v * 100).toFixed(1)}%` : '—';
  const cm = perf?.confusion_matrix;
  const perGenerator = perf?.per_generator ?? [];

  if (unavailable) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            No Evaluation Data
          </CardTitle>
          <CardDescription>
            The backend has not been trained/evaluated yet. Run{' '}
            <code className="font-mono bg-muted px-1 rounded">python -m scripts.train_demo</code>{' '}
            to produce model artifacts and a performance report.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
          <CardDescription>
            Held-out metrics from the persisted evaluation report
            {perf?.generated_at ? ` · generated ${new Date(perf.generated_at).toLocaleString()}` : ''}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="technical" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="technical">Headline Metrics</TabsTrigger>
              <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
              <TabsTrigger value="slices">Slices & Calibration</TabsTrigger>
            </TabsList>

            <TabsContent value="technical" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard icon={Target} title="Accuracy" value={pct(perf?.accuracy ?? model?.accuracy)} />
                <MetricCard icon={Activity} title="ROC AUC" value={pct(perf?.roc_auc ?? model?.roc_auc)} />
                <MetricCard icon={BarChart3} title="F1 Score" value={pct(perf?.f1 ?? model?.f1)} />
                <MetricCard
                  icon={TrendingUp}
                  title="FPR @ Threshold"
                  value={pct(perf?.fpr_at_threshold)}
                  target={perf?.threshold != null ? `threshold ${perf.threshold.toFixed(2)}` : undefined}
                />
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <InfoTile label="Precision" value={pct(perf?.precision)} />
                <InfoTile label="Recall" value={pct(perf?.recall)} />
                <InfoTile
                  label="ECE (calibration error)"
                  value={pct(perf?.ece ?? model?.ece)}
                  note="target < 0.05"
                />
              </div>

              <p className="text-xs text-muted-foreground mt-2">
                Metrics reflect the demo/synthetic training set until a real multi-generator dataset is
                supplied via a manifest. Latency p95 on CPU:{' '}
                {perf?.latency_ms ? `${perf.latency_ms.p95}ms` : '—'} · peak RSS:{' '}
                {perf?.peak_rss_mb != null ? `${perf.peak_rss_mb.toFixed(0)} MB` : '—'} · samples:{' '}
                {perf?.n_samples ?? model?.n_training_samples ?? '—'}
              </p>
            </TabsContent>

            <TabsContent value="confusion" className="space-y-4">
              {!cm ? (
                <EmptyState message="No confusion matrix in the current report." />
              ) : (
                <div className="max-w-sm space-y-2">
                  {[
                    ['True Negatives (real → authentic)', cm.tn],
                    ['False Positives (real → AI)', cm.fp],
                    ['False Negatives (AI → authentic)', cm.fn],
                    ['True Positives (AI → AI)', cm.tp],
                  ].map(([label, value]) => (
                    <div
                      key={String(label)}
                      className="flex justify-between items-center p-3 bg-muted/30 rounded-lg border border-border"
                    >
                      <span className="text-sm text-muted-foreground">{label}</span>
                      <span className="font-bold">{value}</span>
                    </div>
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="slices" className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">Per-generator slices</h3>
                {perGenerator.length === 0 ? (
                  <EmptyState message="No generator labels in the training manifest — supply one with a 'generator' column to enable slice reports." />
                ) : (
                  <div className="space-y-2">
                    {perGenerator.map((row) => (
                      <div
                        key={row.generator}
                        className="flex justify-between items-center p-3 bg-muted/30 rounded-lg border border-border"
                      >
                        <span className="font-medium">{row.generator}</span>
                        <div className="flex gap-6 text-sm">
                          <span>
                            n=<span className="font-bold">{row.n}</span>
                          </span>
                          <span>
                            acc=<span className="font-bold">{(row.accuracy * 100).toFixed(1)}%</span>
                          </span>
                          <span>
                            f1=<span className="font-bold">{(row.f1 * 100).toFixed(1)}%</span>
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div>
                <h3 className="font-semibold mb-2">Latency profile</h3>
                {perf?.latency_ms ? (
                  <div className="grid grid-cols-3 gap-4 max-w-md">
                    <InfoTile label="p50" value={`${perf.latency_ms.p50}ms`} />
                    <InfoTile label="p95" value={`${perf.latency_ms.p95}ms`} />
                    <InfoTile label="p99" value={`${perf.latency_ms.p99}ms`} />
                  </div>
                ) : (
                  <EmptyState message="No latency measurements recorded." />
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Loaded Model</CardTitle>
          <CardDescription>Served artifacts metadata (GET /api/v1/model)</CardDescription>
        </CardHeader>
        <CardContent>
          {model ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <InfoTile label="Version" value={`v${model.version}`} />
              <InfoTile label="Training mode" value={model.training_mode} />
              <InfoTile label="Features" value={model.n_features != null ? String(model.n_features) : '—'} />
              <InfoTile
                label="Trained at"
                value={model.trained_at ? new Date(model.trained_at).toLocaleString() : '—'}
              />
            </div>
          ) : (
            <EmptyState message="No model artifacts loaded." />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  title,
  value,
  target
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  value: string;
  target?: string;
}) {
  return (
    <div className="p-4 bg-muted/30 rounded-lg border border-border">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">{title}</span>
      </div>
      <div className="text-2xl font-bold mb-1">{value}</div>
      {target && <div className="text-xs text-muted-foreground">{target}</div>}
    </div>
  );
}

function InfoTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="p-4 bg-background/40 rounded-lg border border-border">
      <div className="text-sm text-muted-foreground mb-1">{label}</div>
      <div className="text-xl font-bold">{value}</div>
      {note && <div className="text-xs text-muted-foreground">{note}</div>}
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <p className="text-sm text-muted-foreground italic py-4">{message}</p>
  );
}
