import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getPerformance, type PerformanceReport } from "@/lib/api/client";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

/**
 * Model-performance panel (Masterplan §5.4) — renders the persisted held-out
 * evaluation report: accuracy/precision/recall/F1, confusion matrix, ROC +
 * PR curves, ECE, and per-generator breakdown. Reads /api/v1/model/performance.
 */
const ModelPerformance = () => {
  const [report, setReport] = useState<PerformanceReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPerformance()
      .then(setReport)
      .catch((err) => setError(err instanceof Error ? err.message : "No performance report yet."));
  }, []);

  if (error) {
    return (
      <div className="glass rounded-xl p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-2">Model Performance</h3>
        <p className="text-sm text-muted-foreground">Loading held-out evaluation…</p>
      </div>
    );
  }

  const cmData = [
    { name: "Authentic", "Predicted Authentic": report.confusion_matrix.tn, "Predicted AI": report.confusion_matrix.fp },
    { name: "AI", "Predicted Authentic": report.confusion_matrix.fn, "Predicted AI": report.confusion_matrix.tp },
  ];

  const rocData = report.roc_curve?.x?.map((x, i) => ({ fpr: x, tpr: report.roc_curve.y[i] })) ?? [];
  const prData = report.pr_curve?.x?.map((x, i) => ({ recall: x, precision: report.pr_curve.y[i] })) ?? [];

  const meta = [
    { label: "Accuracy", value: `${(report.accuracy * 100).toFixed(1)}%` },
    { label: "Precision", value: `${(report.precision * 100).toFixed(1)}%` },
    { label: "Recall", value: `${(report.recall * 100).toFixed(1)}%` },
    { label: "F1", value: `${(report.f1 * 100).toFixed(1)}%` },
    { label: "ROC AUC", value: report.roc_auc.toFixed(3) },
    { label: "ECE", value: report.ece.toFixed(4) },
    { label: "FPR", value: `${(report.fpr_at_threshold * 100).toFixed(2)}%` },
    { label: "Threshold", value: report.threshold.toFixed(3) },
  ];

  return (
    <div className="glass rounded-xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Model Performance (Held-Out)</h3>
        <Badge variant="outline">generated {new Date(report.generated_at).toLocaleString()}</Badge>
      </div>

      {/* KPI tiles */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {meta.map((m) => (
          <div key={m.label} className="rounded-lg bg-muted/30 p-3">
            <div className="text-xs text-muted-foreground">{m.label}</div>
            <div className="font-mono text-lg font-medium">{m.value}</div>
          </div>
        ))}
      </div>

      {/* Confusion matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Confusion Matrix</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={cmData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Predicted Authentic" fill="hsl(var(--success))" />
                <Bar dataKey="Predicted AI" fill="hsl(var(--destructive))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Per-generator pie */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Per-Generator Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            {report.per_generator?.length ? (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={report.per_generator.map((g) => ({ name: g.generator, value: g.n }))}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {report.per_generator.map((_, i) => (
                      <Cell key={i} fill={`hsl(${(i * 70) % 360}, 70%, 50%)`} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground">No per-generator data.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ROC + PR curves */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">ROC Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottom", offset: -2 }} />
                <YAxis dataKey="tpr" type="number" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line dataKey="tpr" stroke="hsl(var(--primary))" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Precision-Recall Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={prData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} label={{ value: "Recall", position: "insideBottom", offset: -2 }} />
                <YAxis dataKey="precision" type="number" domain={[0, 1]} label={{ value: "Precision", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Line dataKey="precision" stroke="hsl(var(--accent))" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ModelPerformance;
