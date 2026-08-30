import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, AlertTriangle, Download, Sparkles, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import ConfidenceMeter from "./ConfidenceMeter";
import FrequencySpectrum from "./FrequencySpectrum";
import WaveletVisualization from "./WaveletVisualization";
import ModelAttribution from "./ModelAttribution";
import HeatmapOverlay from "./HeatmapOverlay";
import type { AdaptiveAnalysisResult } from "@/lib/services/adaptiveInference";

interface AnalysisResultsProps {
  isVisible: boolean;
  result: AdaptiveAnalysisResult | null;
  /** Data-URL preview of the uploaded image for the saliency overlay. */
  previewUrl?: string | null;
}

const AnalysisResults = ({ isVisible, result, previewUrl = null }: AnalysisResultsProps) => {
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const hasResult = isVisible && !!result;
  const raw = result?.raw ?? null;

  const sortedParameters = useMemo(
    () => (result ? [...result.forensicParameters].sort((a, b) => b.score - a.score) : []),
    [result]
  );

  useEffect(() => {
    setOverlayOpacity(0.65);
  }, [raw?.request_id]);

  const getResultIcon = () => {
    if (!result) return null;
    const tone = result.verdict?.tone;
    if (tone === "warning") {
      return <AlertTriangle className="h-8 w-8 text-warning" />;
    }
    return tone === "ai" ? (
      <XCircle className="h-8 w-8 text-destructive" />
    ) : (
      <CheckCircle className="h-8 w-8 text-success" />
    );
  };

  const getResultText = () => {
    if (!result) return "";
    return result.verdict?.label ?? "Uncertain";
  };

  const getResultColor = (): "real" | "ai" | "warning" => {
    if (!result) return "warning";
    const tone = result.verdict?.tone;
    if (tone === "warning") return "warning";
    return tone === "ai" ? "ai" : "real";
  };

  const handleDownloadReport = () => {
    if (!raw) return;
    // Export the full JSON payload (verdict + all measured features).
    const blob = new Blob([JSON.stringify(raw, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `frequency_guard_${raw.request_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!hasResult || !result || !raw) return null;

  return (
    <motion.section
      className="py-12"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          {/* FrequencyGuard Phase 4: NLP narrative card */}
          {raw.explanation && (
            <motion.div
              className="glass rounded-2xl p-6 mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Linguistic Summary</h3>
              </div>
              <div className="flex items-center gap-2 mb-3">
                <Badge variant="outline" className="font-mono">
                  {raw.explanation.severity}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {raw.explanation.sentence_count} sentence{raw.explanation.sentence_count === 1 ? "" : "s"}
                </span>
              </div>
              <p className="text-foreground leading-relaxed text-sm">{raw.explanation.narrative}</p>
            </motion.div>
          )}

          {/* FrequencyGuard Phase 3: semantic-evidence card */}
          {raw.semantic && raw.semantic.triggered && (
            <motion.div
              className="glass rounded-2xl p-6 mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.15 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle
                  className={`h-5 w-5 ${raw.semantic.suspicious ? "text-warning" : "text-muted-foreground"}`}
                />
                <h3 className="text-lg font-semibold">Semantic Arbitration</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">{raw.semantic.summary}</p>
              {raw.semantic.flags.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {raw.semantic.flags.map((flag, idx) => (
                    <div key={idx} className="rounded-lg border border-border bg-muted/20 p-3">
                      <div className="flex items-center justify-between mb-1">
                        <Badge variant="outline" className="capitalize">{flag.node}</Badge>
                        <span className="text-xs font-mono text-muted-foreground">
                          {(flag.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-sm text-foreground">{flag.label}</p>
                      {flag.bbox && (
                        <p className="text-xs text-muted-foreground mt-1 font-mono">
                          bbox [x,y,w,h]: {flag.bbox.join(", ")}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No detectable semantic anomalies.</p>
              )}
            </motion.div>
          )}

          {/* Primary Result Card */}
          <motion.div
            className="glass rounded-2xl p-8 mb-8"
            initial={{ scale: 0.95 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-6">
                <div className={`
                  w-20 h-20 rounded-2xl flex items-center justify-center
                  ${result.verdict?.tone === "ai" ? "bg-destructive/20" : result.verdict?.tone === "real" ? "bg-success/20" : "bg-warning/20"}
                `}>
                  {getResultIcon()}
                </div>
                <div>
                  <h2 className="text-2xl font-bold mb-1">{getResultText()}</h2>
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-muted-foreground">
                      {result.verdict?.state === "uncertain"
                        ? result.verdict.detail
                        : result.isAI
                          ? `Attributed to ${result.model}`
                          : "No AI generation patterns detected"}
                    </p>
                    <Badge variant="secondary">{result.model}</Badge>
                    <Badge variant="outline">threshold {raw.threshold.toFixed(2)}</Badge>
                    <Badge variant="outline">{raw.latency_ms.toFixed(0)} ms</Badge>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="w-48">
                  <ConfidenceMeter
                    value={result.confidence}
                    label="Confidence"
                    color={getResultColor()}
                  />
                </div>
                <div className="flex gap-2">
                  <Button variant="glass" size="icon" onClick={handleDownloadReport} title="Download JSON report">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Detailed Analysis Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FrequencySpectrum isActive={isVisible} spectrum={raw.spectrum} azimuthal={raw.azimuthal} />
            <WaveletVisualization isActive={isVisible} bands={raw.wavelet_bands} />
            <div className="lg:col-span-2">
              <ModelAttribution
                isActive={isVisible}
                families={raw.families}
                attributionMode={raw.attribution_mode}
              />
            </div>
          </div>

          {/* Explainability Overlay */}
          <motion.div
            className="glass rounded-2xl p-6 mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Layers className="h-5 w-5 text-primary" />
                  Spectral-Saliency Overlay
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Tiles whose local spectrum deviates most from the natural 1/f law glow yellow —
                  synthetic content concentrates there. Patch-inconsistency score:{" "}
                  <span className="font-mono">{raw.explainability.patch_inconsistency_score.toFixed(3)}</span>
                </p>
              </div>
              <div className="w-full sm:w-56">
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Overlay opacity</span>
                  <span>{Math.round(overlayOpacity * 100)}%</span>
                </div>
                <Slider
                  value={[overlayOpacity * 100]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={(v) => setOverlayOpacity(v[0] / 100)}
                />
              </div>
            </div>
            <HeatmapOverlay
              preview={previewUrl}
              heatmapSrc={raw.explainability.saliency_png_base64}
              opacity={overlayOpacity}
            />
          </motion.div>

          {/* Feature Analysis */}
          <motion.div
            className="glass rounded-2xl p-6 mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-1">Forensic Feature Analysis</h3>
            <p className="text-xs text-muted-foreground mb-4">
              Every value below is measured from your image by the frequency-domain pipeline.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              {sortedParameters.map((feature) => (
                <div key={feature.label} className="bg-muted/30 rounded-lg p-4" title={feature.description}>
                <div className="text-sm text-muted-foreground mb-1">{feature.label}</div>
                  <div className={`text-lg font-mono font-semibold ${feature.score > 0.65 ? "text-destructive" : "text-success"}`}>
                    {feature.value}
                  </div>
                  <div className="mt-2 h-1.5 rounded bg-muted overflow-hidden">
                    <div className="h-full rounded bg-primary" style={{ width: `${feature.score * 100}%` }} />
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">{feature.description}</p>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            className="glass rounded-2xl p-6 mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  Provenance
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Request <span className="font-mono">{raw.request_id}</span> · model{" "}
                  <span className="font-mono">v{raw.model_version}</span> · sha256{" "}
                  <span className="font-mono text-xs">{raw.sha256.slice(0, 16)}…</span> · source{" "}
                  {raw.image_size[1]}×{raw.image_size[0]}px
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
};

export default AnalysisResults;
