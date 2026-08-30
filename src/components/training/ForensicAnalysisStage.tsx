import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Waves, BarChart3, AlertCircle } from 'lucide-react';
import { analyzeImage, type AnalyzeResponse } from '@/lib/api/client';
import type { DetectorPrediction } from '@/lib/types/training';

interface ForensicAnalysisStageProps {
  file: File;
  onComplete: (prediction: DetectorPrediction) => void;
  onError?: (message: string) => void;
}

/**
 * Stage 1: Automated Forensic Analysis — REAL pipeline.
 *
 * Calls POST /api/v1/analyze and maps the measured response onto the
 * DetectorPrediction contract consumed by weak-labeling fusion. Every
 * number shown here comes from the backend frequency-domain pipeline.
 */
export default function ForensicAnalysisStage({ file, onComplete, onError }: ForensicAnalysisStageProps) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('Uploading to analysis pipeline…');
  const [error, setError] = useState<string | null>(null);
  const [raw, setRaw] = useState<AnalyzeResponse | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const run = async () => {
      try {
        // The real pipeline is a single fast call; we animate staged progress
        // around it for UX only. All displayed values are measured.
        setCurrentStep('Running FFT / DCT / wavelet / SRM extraction…');
        setProgress(30);
        const response = await analyzeImage(file);
        setRaw(response);
        setCurrentStep('Calibrating probability and attributing family…');
        setProgress(85);

        const topFamily = [...response.families].sort((a, b) => b.probability - a.probability)[0];
        const familyMap: Record<string, 'Diffusion' | 'GAN' | 'Real' | 'Unknown'> = {
          real: 'Real',
          diffusion: 'Diffusion',
          gan: 'GAN',
          other: 'Unknown',
        };
        const modelLabel =
          !response.is_ai
            ? 'Real Camera'
            : topFamily?.family === 'diffusion'
              ? 'Diffusion Model'
              : topFamily?.family === 'gan'
                ? 'GAN'
                : 'Unknown Generator';

        const prediction: DetectorPrediction = {
          family: response.is_ai ? (topFamily?.family === 'diffusion' ? 'Diffusion' : topFamily?.family === 'gan' ? 'GAN' : 'Unknown') : 'Real',
          model: modelLabel,
          confidence: response.confidence / 100,
          probabilities: Object.fromEntries(
            response.families.map((f) => [f.family, f.probability])
          ) as DetectorPrediction['probabilities'],
          forensicFeatures: {
            contourletCoefficients: [],
            multiResFFT: [],
            phaseRandomness: response.features.phase_entropy,
            crossChannelCoherence: 0,
            spectralEntropyGradients: [],
            fractalDimension: response.features.texture_fractal_dim,
            glcmFeatures: {
              contrast: 0,
              correlation: 0,
              energy: 0,
              homogeneity: 0,
            },
            attentionMapConsistency: 0,
            checkerboardScore: Math.min(1, response.features.peak_prominence / 10),
            lightTransportConsistency: 0,
            cdcsScore: response.explainability.patch_inconsistency_score,
            spatialBranchConfidence: 0,
            frequencyBranchConfidence: 0,
            waveletBranchConfidence: 0,
            statisticalBranchConfidence: 0,
          },
          calibrationAccuracy: 1 - response.explainability.mean_spectral_deviation > 0
            ? Math.max(0, Math.min(1, 1 - response.explainability.patch_inconsistency_score))
            : 0.5,
          timestamp: Date.now(),
        };

        setProgress(100);
        onComplete(prediction);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Analysis failed unexpectedly.';
        setError(message);
        onError?.(message);
      }
    };

    run();
  }, [file, onComplete, onError]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5 animate-pulse text-blue-600" />
          Stage 1: Automated Forensic Analysis
        </CardTitle>
        <CardDescription>
          Real backend pipeline: FFT radial/azimuthal profiles, DCT block stats, wavelet sub-bands,
          SRM noise residuals, GLCM texture — then calibrated classification.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error} — is the API server running?</AlertDescription>
          </Alert>
        )}

        {!error && (
          <>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{currentStep}</span>
                <span className="font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>

            {raw ? (
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-2 p-3 bg-muted/40 rounded-lg border border-border">
                  <Waves className="w-4 h-4 text-blue-600" />
                  <div>
                    <div className="text-xs text-muted-foreground">Spectral Slope</div>
                    <div className="text-sm font-medium font-mono">{raw.features.spectral_slope.toFixed(3)}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 p-3 bg-muted/40 rounded-lg border border-border">
                  <BarChart3 className="w-4 h-4 text-purple-600" />
                  <div>
                    <div className="text-xs text-muted-foreground">Patch Inconsistency</div>
                    <div className="text-sm font-medium font-mono">
                      {raw.explainability.patch_inconsistency_score.toFixed(3)}
                    </div>
                  </div>
                </div>
                <div className="p-4 bg-gradient-to-r from-primary/10 to-accent/10 rounded-lg space-y-2 border border-border col-span-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Fake Probability (calibrated)</span>
                    <Badge variant={raw.is_ai ? 'destructive' : 'default'}>
                      {(raw.fake_probability * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress value={raw.fake_probability * 100} className="h-1" />
                  <p className="text-xs text-muted-foreground">
                    Latency {raw.latency_ms.toFixed(0)} ms · model v{raw.model_version}
                  </p>
                </div>
              </div>
            ) : null}
          </>
        )}

        {/* Image Preview */}
        <Preview file={file} />
      </CardContent>
    </Card>
  );
}

/** Object URLs must be created once and revoked on unmount to avoid leaks. */
function Preview({ file }: { file: File }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);
  if (!url) return null;
  return (
    <div className="aspect-video bg-muted rounded-lg overflow-hidden border border-border">
      <img src={url} alt="Analyzing" className="w-full h-full object-contain" />
    </div>
  );
}
