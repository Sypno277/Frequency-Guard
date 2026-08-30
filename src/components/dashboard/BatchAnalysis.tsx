import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Upload, Download, X, Loader2 } from "lucide-react";
import {
  submitBatch,
  pollJob,
  type BatchJobResponse,
  type BatchImageResult,
} from "@/lib/api/client";
import { downloadBatchReportPdf } from "@/lib/services/pdfReport";

interface BatchAnalysisProps {
  /** Per-image analysis result callback (used to seed history refresh). */
  onJobComplete?: (job: BatchJobResponse) => void;
}

/**
 * Batch analysis view (Masterplan §5.2).
 *
 * Multi-file upload → POST /api/v1/batch → poll /api/v1/jobs/{id} → render
 * incremental results table. Download button exports the CSV job report.
 * Also exposes the raw JSON job result for the optional PDF export path.
 */
const BatchAnalysis = ({ onJobComplete }: BatchAnalysisProps) => {
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<BatchJobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const images = Array.from(incoming).filter((f) => f.type.startsWith("image/"));
    setFiles((prev) => [...prev, ...images].slice(0, 64));
    setJob(null);
    setError(null);
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setFiles([]);
    setJob(null);
    setError(null);
  };

  const handleRun = async () => {
    if (!files.length) return;
    setIsSubmitting(true);
    setError(null);
    setJob(null);
    try {
      const jobId = await submitBatch(files);
      const result = await pollJob(jobId.job_id, (snap) => setJob({ ...snap }));
      setJob(result);
      onJobComplete?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch job failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCsvDownload = (jobId: string) => {
    window.open(`${import.meta.env.VITE_API_BASE ?? ""}/api/v1/jobs/${jobId}/csv`, "_blank");
  };

  const done = files.filter((f) => f.size > 0).length;
  const jobResults = job?.results ?? [];
  const doneCount = jobResults.filter((r) => r.status === "done").length;
  const errorCount = jobResults.filter((r) => r.status === "error").length;

  return (
    <div className="glass rounded-xl p-6 space-y-6">
      <div>
        <h3 className="text-lg font-semibold">Batch Analysis</h3>
        <p className="text-sm text-muted-foreground">
          Upload up to 64 images; the API processes them asynchronously on a worker pool. Results
          stream in as each image finishes.
        </p>
      </div>

      {/* Dropzone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer hover:border-primary/50 ${
          isSubmitting ? "opacity-60 pointer-events-none" : ""
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <Upload className="h-10 w-10 mx-auto mb-3 text-primary" />
        <p className="font-medium">Drop images or click to browse</p>
        <p className="text-sm text-muted-foreground mt-1">
          {files.length ? `${files.length} selected` : "JPG, PNG, WebP, BMP"}
        </p>
      </div>

      {/* Selected files */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((f, i) => (
            <Badge key={`${f.name}-${i}`} variant="secondary" className="gap-1 pr-1">
              {f.name}
              <Button
                variant="ghost"
                size="icon"
                className="h-4 w-4 rounded-full"
                onClick={() => removeFile(i)}
                disabled={isSubmitting}
                aria-label={`Remove ${f.name}`}
              >
                <X className="h-3 w-3" />
              </Button>
            </Badge>
          ))}
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex gap-2">
        <Button onClick={handleRun} disabled={!files.length || isSubmitting} className="flex-1">
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Analyzing…
            </>
          ) : (
            "Run Batch Analysis"
          )}
        </Button>
        <Button variant="outline" onClick={clearAll} disabled={isSubmitting}>
          Clear
        </Button>
      </div>

      {/* Progress + Results */}
      {job && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              Progress: {job.completed}/{job.total_images} · {doneCount} done · {errorCount} errors
            </span>
            <Badge variant={job.status === "completed" ? "default" : "secondary"}>{job.status}</Badge>
          </div>

          {jobResults.length > 0 && (
            <>
              <div className="rounded-lg border border-border overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-muted/30 text-left">
                    <tr>
                      <th className="px-3 py-2">#</th>
                      <th className="px-3 py-2">Filename</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Verdict</th>
                      <th className="px-3 py-2">Fake Prob</th>
                      <th className="px-3 py-2">Family</th>
                      <th className="px-3 py-2">ms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobResults
                      .slice()
                      .sort((a, b) => a.index - b.index)
                      .map((r: BatchImageResult) => (
                        <tr key={`${r.index}-${r.filename}`} className="border-t border-border">
                          <td className="px-3 py-2">{r.index}</td>
                          <td className="px-3 py-2 max-w-[220px] truncate">{r.filename}</td>
                          <td className="px-3 py-2">{r.status}</td>
                          <td className="px-3 py-2">
                            {r.status === "error" ? (
                              <span className="text-destructive">{r.error ?? "error"}</span>
                            ) : r.is_ai ? (
                              <Badge variant="destructive">AI</Badge>
                            ) : (
                              <Badge variant="secondary">Authentic</Badge>
                            )}
                          </td>
                          <td className="px-3 py-2 font-mono">
                            {r.fake_probability != null ? (r.fake_probability * 100).toFixed(1) + "%" : "—"}
                          </td>
                          <td className="px-3 py-2">{r.family ?? "—"}</td>
                          <td className="px-3 py-2 font-mono">{r.latency_ms?.toFixed(0) ?? "—"}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              <div className="flex gap-2">
                <Button variant="outline" onClick={() => handleCsvDownload(job.job_id)}>
                  <Download className="h-4 w-4" /> Download CSV
                </Button>
                <Button variant="outline" onClick={() => downloadBatchReportPdf(job)}>
                  <Download className="h-4 w-4" /> Download PDF
                </Button>
                <Button variant="outline" onClick={() => {
                  const blob = new Blob([JSON.stringify(job, null, 2)], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `batch_${job.job_id}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}>
                  Export JSON
                </Button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default BatchAnalysis;
