import { useEffect, useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Database, TrendingUp, Layers, Activity } from 'lucide-react';
import { getModelInfo, getPerformance, type ModelInfo, type PerformanceReport } from '@/lib/api/client';
import TrainingIntake from '@/components/training/TrainingIntake';
import TrainingDashboard from '@/components/training/TrainingDashboard';
import ModelManagement from '@/components/training/ModelManagement';
import EmbeddingVisualizer from '@/components/training/EmbeddingVisualizer';

export default function Training() {
  const [activeTab, setActiveTab] = useState('intake');
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [perf, setPerf] = useState<PerformanceReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo().then(setModel).catch(() => undefined);
    getPerformance().then(setPerf).catch(() => undefined);
  }, []);

  const accuracy = model?.accuracy ?? perf?.accuracy;
  const nSamples = model?.n_training_samples ?? perf?.n_samples;
  const ece = model?.ece ?? perf?.ece;
  const modelAvailable = model !== null;

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 gradient-text">
            Model Training Center
          </h1>
          <p className="text-muted-foreground">
            Expandable training system with weak signal learning, embedding-based clustering, and dynamic model addition
          </p>
        </div>

        {!modelAvailable && (
          <Alert variant="destructive" className="mb-8">
            <Activity className="h-4 w-4" />
            <AlertDescription>
              No trained model artifacts found. Run{' '}
              <code className="font-mono bg-muted px-1 rounded">python -m scripts.train_demo</code>{' '}
              or supply a manifest first. Metrics below show demo-state values until then.
            </AlertDescription>
          </Alert>
        )}

        {/* Quick Stats — real values from the API when a model is loaded */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Training Samples</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{nSamples ?? 0}</div>
              <p className="text-xs text-muted-foreground">
                {model?.training_mode ?? 'no model loaded'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Model Version</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">v{model?.version ?? '—'}</div>
              <p className="text-xs text-muted-foreground">
                {model?.trained_at ? new Date(model.trained_at).toLocaleString() : 'not trained'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—'}
              </div>
              <p className="text-xs text-muted-foreground">
                {ece != null ? `ECE ${(ece * 100).toFixed(2)}%` : 'not evaluated'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Features</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{model?.n_features ?? '—'}</div>
              <p className="text-xs text-muted-foreground">dim feature vector</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Training Interface */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="intake" className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Training Intake
            </TabsTrigger>
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="models" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Model Management
            </TabsTrigger>
            <TabsTrigger value="embeddings" className="flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Embeddings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="intake" className="space-y-4">
            <TrainingIntake />
          </TabsContent>

          <TabsContent value="dashboard" className="space-y-4">
            <TrainingDashboard />
          </TabsContent>

          <TabsContent value="models" className="space-y-4">
            <ModelManagement />
          </TabsContent>

          <TabsContent value="embeddings" className="space-y-4">
            <EmbeddingVisualizer />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
