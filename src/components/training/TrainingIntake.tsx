import { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import ForensicAnalysisStage from '@/components/training/ForensicAnalysisStage';
import WeakLabelingStage from '@/components/training/WeakLabelingStage';
import type { DetectorPrediction, UserLabel, TrainingSignal } from '@/lib/types/training';

type IntakeStage = 'upload' | 'analyzing' | 'labeling' | 'complete';

export default function TrainingIntake() {
  const [stage, setStage] = useState<IntakeStage>('upload');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [detectorPrediction, setDetectorPrediction] = useState<DetectorPrediction | null>(null);
  const [trainingSignals, setTrainingSignals] = useState<TrainingSignal[]>([]);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(acceptedFiles);
    setStage('analyzing');
    setCurrentFileIndex(0);
    setProgress(0);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    multiple: true
  });

  const handleAnalysisComplete = (prediction: DetectorPrediction) => {
    setDetectorPrediction(prediction);
    setStage('labeling');
  };

  const handleLabelSubmit = (userLabel: UserLabel | null) => {
    if (!detectorPrediction) return;

    // Create training signal with weak signal fusion
    const trainingSignal: TrainingSignal = {
      imageId: `img_${Date.now()}_${currentFileIndex}`,
      detectorPrediction,
      userLabel,
      fusedLabel: fuseSignals(detectorPrediction, userLabel),
      embeddingVector: generateEmbedding(detectorPrediction),
      status: userLabel ? 'labeled' : 'pending_label'
    };

    setTrainingSignals([...trainingSignals, trainingSignal]);

    // Move to next file or complete
    if (currentFileIndex < uploadedFiles.length - 1) {
      setCurrentFileIndex(currentFileIndex + 1);
      setStage('analyzing');
      setProgress(((currentFileIndex + 1) / uploadedFiles.length) * 100);
    } else {
      setStage('complete');
      setProgress(100);
    }
  };

  const handleSkipLabel = () => {
    handleLabelSubmit(null);
  };

  const resetIntake = () => {
    setStage('upload');
    setUploadedFiles([]);
    setCurrentFileIndex(0);
    setDetectorPrediction(null);
    setProgress(0);
  };

  // Helper: Fuse detector and user signals
  const fuseSignals = (
    detector: DetectorPrediction,
    user: UserLabel | null
  ): TrainingSignal['fusedLabel'] => {
    if (!user) {
      return {
        family: detector.family,
        model: detector.model,
        confidence: detector.confidence,
        detectorWeight: 1.0,
        userWeight: 0.0
      };
    }

    // Weight by calibration accuracy and user confidence
    const detectorWeight = detector.calibrationAccuracy * 0.7;
    const userWeight = (user.confidence / 100) * (user.userAccuracy || 0.5) * 0.3;
    const totalWeight = detectorWeight + userWeight;

    // Normalize weights
    const normDetectorWeight = detectorWeight / totalWeight;
    const normUserWeight = userWeight / totalWeight;

    // Choose model based on higher weighted confidence
    const detectorScore = detector.confidence * normDetectorWeight;
    const userScore = (user.confidence / 100) * normUserWeight;

    if (detectorScore > userScore) {
      return {
        family: detector.family,
        model: detector.model,
        confidence: detectorScore + userScore,
        detectorWeight: normDetectorWeight,
        userWeight: normUserWeight
      };
    } else {
      return {
        family: user.family,
        model: user.model,
        confidence: detectorScore + userScore,
        detectorWeight: normDetectorWeight,
        userWeight: normUserWeight
      };
    }
  };

  // Placeholder for embedding generation
  const generateEmbedding = (prediction: DetectorPrediction): number[] => {
    // In production: Use ViT/ConvNeXt backbone to generate 512/1024-dim vector
    return new Array(512).fill(0).map(() => Math.random());
  };

  return (
    <div className="space-y-6">
      {/* Upload Stage */}
      {stage === 'upload' && (
        <Card>
          <CardHeader>
            <CardTitle>Upload Training Images</CardTitle>
            <CardDescription>
              Two-stage intake pipeline: Automated forensic analysis → Optional weak labeling
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-muted-foreground/40'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              {isDragActive ? (
                <p className="text-primary font-medium">Drop images here...</p>
              ) : (
                <>
                  <p className="text-lg font-medium mb-2">
                    Drag & drop training images here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Supports PNG, JPG, JPEG, WebP
                  </p>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Progress Bar */}
      {stage !== 'upload' && uploadedFiles.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing: {currentFileIndex + 1} / {uploadedFiles.length}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stage 1: Forensic Analysis */}
      {stage === 'analyzing' && uploadedFiles[currentFileIndex] && (
        <ForensicAnalysisStage
          file={uploadedFiles[currentFileIndex]}
          onComplete={handleAnalysisComplete}
        />
      )}

      {/* Stage 2: Weak User Labeling */}
      {stage === 'labeling' && detectorPrediction && (
        <WeakLabelingStage
          file={uploadedFiles[currentFileIndex]}
          detectorPrediction={detectorPrediction}
          onSubmit={handleLabelSubmit}
          onSkip={handleSkipLabel}
        />
      )}

      {/* Complete Stage */}
      {stage === 'complete' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              Training Intake Complete
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <AlertDescription>
                Successfully processed {uploadedFiles.length} images. {trainingSignals.filter(s => s.userLabel).length} images labeled by user.
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
                <div className="text-2xl font-bold">{trainingSignals.length}</div>
                <div className="text-sm text-muted-foreground">Training Signals Generated</div>
              </div>
              <div className="p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                <div className="text-2xl font-bold">
                  {trainingSignals.filter(s => s.userLabel).length}
                </div>
                <div className="text-sm text-muted-foreground">User Labels Provided</div>
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={resetIntake} className="flex-1">
                Upload More Images
              </Button>
              <Button variant="outline" className="flex-1">
                View Training Signals
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
