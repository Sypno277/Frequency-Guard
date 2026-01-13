import { motion } from "framer-motion";
import { CheckCircle, XCircle, AlertTriangle, Download, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import ConfidenceMeter from "./ConfidenceMeter";
import FrequencySpectrum from "./FrequencySpectrum";
import WaveletVisualization from "./WaveletVisualization";
import ModelAttribution from "./ModelAttribution";

interface AnalysisResultsProps {
  isVisible: boolean;
  result: {
    isAI: boolean;
    confidence: number;
    model: string;
  } | null;
}

const AnalysisResults = ({ isVisible, result }: AnalysisResultsProps) => {
  if (!isVisible || !result) return null;

  const getResultIcon = () => {
    if (result.confidence < 50) {
      return <AlertTriangle className="h-8 w-8 text-warning" />;
    }
    return result.isAI ? (
      <XCircle className="h-8 w-8 text-destructive" />
    ) : (
      <CheckCircle className="h-8 w-8 text-success" />
    );
  };

  const getResultText = () => {
    if (result.confidence < 50) {
      return "Uncertain";
    }
    return result.isAI ? "AI Generated" : "Authentic Image";
  };

  const getResultColor = (): "real" | "ai" | "warning" => {
    if (result.confidence < 50) return "warning";
    return result.isAI ? "ai" : "real";
  };

  return (
    <motion.section
      className="py-12"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
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
                  ${result.isAI ? "bg-destructive/20" : result.confidence < 50 ? "bg-warning/20" : "bg-success/20"}
                `}>
                  {getResultIcon()}
                </div>
                <div>
                  <h2 className="text-2xl font-bold mb-1">{getResultText()}</h2>
                  <p className="text-muted-foreground">
                    {result.isAI
                      ? `Detected as ${result.model} output`
                      : "No AI generation patterns detected"}
                  </p>
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
                  <Button variant="glass" size="icon">
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button variant="glass" size="icon">
                    <Share2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Detailed Analysis Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FrequencySpectrum isActive={isVisible} />
            <WaveletVisualization isActive={isVisible} />
            <div className="lg:col-span-2">
              <ModelAttribution isActive={isVisible} detectedModel={result.isAI ? result.model : "Real"} />
            </div>
          </div>

          {/* Feature Analysis */}
          <motion.div
            className="glass rounded-2xl p-6 mt-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-4">Forensic Feature Analysis</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { label: "High-Freq Anomaly", value: result.isAI ? "0.847" : "0.123", alert: result.isAI },
                { label: "Spectral Flatness", value: result.isAI ? "0.312" : "0.687", alert: result.isAI },
                { label: "Phase Coherence", value: result.isAI ? "0.923" : "0.456", alert: result.isAI },
                { label: "Noise Signature", value: result.isAI ? "Synthetic" : "Natural", alert: result.isAI },
              ].map((feature) => (
                <div key={feature.label} className="bg-muted/30 rounded-lg p-4">
                  <div className="text-sm text-muted-foreground mb-1">{feature.label}</div>
                  <div className={`text-lg font-mono font-semibold ${feature.alert ? "text-destructive" : "text-success"}`}>
                    {feature.value}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </motion.section>
  );
};

export default AnalysisResults;
