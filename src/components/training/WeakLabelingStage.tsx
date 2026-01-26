import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { UserCircle, ChevronRight, SkipForward, Info } from 'lucide-react';
import type { DetectorPrediction, UserLabel, ModelFamily, KnownModel } from '@/lib/types/training';

interface WeakLabelingStageProps {
  file: File;
  detectorPrediction: DetectorPrediction;
  onSubmit: (label: UserLabel | null) => void;
  onSkip: () => void;
}

const MODEL_FAMILIES: ModelFamily[] = ['Diffusion', 'GAN', 'Transformer', 'Real', 'Unknown'];

const KNOWN_MODELS: Record<ModelFamily, KnownModel[]> = {
  Diffusion: ['Stable Diffusion XL', 'Stable Diffusion v2.1', 'Stable Diffusion v1.5', 'Custom'],
  GAN: ['StyleGAN3', 'StyleGAN2', 'Custom'],
  Transformer: ['DALL-E 3', 'DALL-E 2', 'Custom'],
  Real: [],
  Unknown: []
};

export default function WeakLabelingStage({ 
  file, 
  detectorPrediction, 
  onSubmit, 
  onSkip 
}: WeakLabelingStageProps) {
  const [selectedFamily, setSelectedFamily] = useState<ModelFamily>(detectorPrediction.family);
  const [selectedModel, setSelectedModel] = useState<string>(detectorPrediction.model);
  const [customModelName, setCustomModelName] = useState('');
  const [confidence, setConfidence] = useState([85]);

  const handleSubmit = () => {
    const userLabel: UserLabel = {
      family: selectedFamily,
      model: selectedModel === 'Custom' ? customModelName : selectedModel,
      confidence: confidence[0],
      userAccuracy: 0.82, // Placeholder: would come from user history
      timestamp: Date.now()
    };
    onSubmit(userLabel);
  };

  const handleFamilyChange = (family: ModelFamily) => {
    setSelectedFamily(family);
    // Reset model selection when family changes
    const models = KNOWN_MODELS[family];
    setSelectedModel(models.length > 0 ? models[0] : '');
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <UserCircle className="w-5 h-5 text-purple-600" />
          Stage 2: Weak User Labeling (Optional)
        </CardTitle>
        <CardDescription>
          Provide a probabilistic label to improve model training. Your label will be weighted against detector prediction.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Detector Prediction Summary */}
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-1">
              <div className="font-medium">Detector Prediction:</div>
              <div className="flex gap-2 items-center">
                <Badge variant="secondary">{detectorPrediction.family}</Badge>
                <Badge>{detectorPrediction.model}</Badge>
                <span className="text-sm text-muted-foreground">
                  {(detectorPrediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                CDCS Score: {(detectorPrediction.forensicFeatures.cdcsScore * 100).toFixed(1)}% | 
                Calibration Accuracy: {(detectorPrediction.calibrationAccuracy * 100).toFixed(1)}%
              </div>
            </div>
          </AlertDescription>
        </Alert>

        {/* Image Preview */}
        <div className="aspect-video bg-muted rounded-lg overflow-hidden border border-border">
          <img
            src={URL.createObjectURL(file)}
            alt="Training sample"
            className="w-full h-full object-contain"
          />
        </div>

        {/* User Label Input */}
        <div className="space-y-4 p-4 bg-primary/5 rounded-lg border border-primary/20">
          <div className="space-y-2">
            <Label>Model Family</Label>
            <Select value={selectedFamily} onValueChange={handleFamilyChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODEL_FAMILIES.map(family => (
                  <SelectItem key={family} value={family}>
                    {family}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {KNOWN_MODELS[selectedFamily].length > 0 && (
            <div className="space-y-2">
              <Label>Specific Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {KNOWN_MODELS[selectedFamily].map(model => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {selectedModel === 'Custom' && (
            <div className="space-y-2">
              <Label>Custom Model Name</Label>
              <Input
                placeholder="e.g., Stable Diffusion XL v1.0 Fine-tuned"
                value={customModelName}
                onChange={(e) => setCustomModelName(e.target.value)}
              />
            </div>
          )}

          <div className="space-y-3">
            <div className="flex justify-between">
              <Label>Your Confidence Level</Label>
              <span className="text-sm font-medium">{confidence[0]}%</span>
            </div>
            <Slider
              value={confidence}
              onValueChange={setConfidence}
              min={0}
              max={100}
              step={5}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Higher confidence gives your label more weight in training.
              Your historical labeling accuracy: 82%
            </p>
          </div>
        </div>

        {/* Signal Fusion Preview */}
        <div className="p-4 bg-gradient-to-r from-primary/10 to-accent/10 rounded-lg space-y-2 border border-border">
          <div className="text-sm font-medium">Weighted Signal Fusion</div>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <div className="text-muted-foreground">Detector Weight</div>
              <div className="font-medium">
                {(detectorPrediction.calibrationAccuracy * 0.7 * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-muted-foreground">User Weight</div>
              <div className="font-medium">
                {((confidence[0] / 100) * 0.82 * 0.3 * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Final training signal will combine both predictions using these weights
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button
            onClick={onSkip}
            variant="outline"
            className="flex-1 flex items-center gap-2"
          >
            <SkipForward className="w-4 h-4" />
            Skip Labeling
          </Button>
          <Button
            onClick={handleSubmit}
            className="flex-1 flex items-center gap-2"
            disabled={selectedModel === 'Custom' && !customModelName.trim()}
          >
            Submit Label
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center">
          Skipping will use detector prediction only. Your contribution helps improve model accuracy.
        </p>
      </CardContent>
    </Card>
  );
}
