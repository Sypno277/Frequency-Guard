import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart3, TrendingUp, Target, Activity } from 'lucide-react';

export default function TrainingDashboard() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Training Performance Metrics</CardTitle>
          <CardDescription>
            Real-time monitoring of model accuracy, AUROC, and calibration metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="technical" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="technical">Technical Metrics</TabsTrigger>
              <TabsTrigger value="permodel">Per-Model Analysis</TabsTrigger>
              <TabsTrigger value="calibration">Calibration</TabsTrigger>
            </TabsList>

            <TabsContent value="technical" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                  icon={Target}
                  title="Overall Accuracy"
                  value="96.4%"
                  change="+2.3%"
                  positive
                  target="≥95%"
                />
                <MetricCard
                  icon={Activity}
                  title="Unknown AI AUROC"
                  value="82.7%"
                  change="+4.1%"
                  positive
                  target="≥80%"
                />
                <MetricCard
                  icon={BarChart3}
                  title="Inference Time"
                  value="1.8s"
                  change="-0.3s"
                  positive
                  target="<2s"
                />
                <MetricCard
                  icon={TrendingUp}
                  title="False Attribution"
                  value="7.2%"
                  change="-1.5%"
                  positive
                  target="≤10%"
                />
              </div>

              <div className="mt-6 p-6 bg-gradient-to-r from-primary/10 to-accent/10 rounded-lg border border-border">
                <h3 className="font-semibold mb-4">Training Progress</h3>
                <div className="space-y-3">
                  <ProgressBar label="Known Models" value={95} color="blue" />
                  <ProgressBar label="Unknown Detection" value={83} color="purple" />
                  <ProgressBar label="Weak Signal Fusion" value={88} color="green" />
                  <ProgressBar label="CDCS Calibration" value={91} color="amber" />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="permodel" className="space-y-4">
              <div className="space-y-3">
                <ModelMetricRow
                  model="Stable Diffusion XL"
                  accuracy={97.2}
                  precision={96.8}
                  recall={97.5}
                  f1={97.1}
                  samples={342}
                />
                <ModelMetricRow
                  model="DALL-E 3"
                  accuracy={95.8}
                  precision={94.9}
                  recall={96.2}
                  f1={95.5}
                  samples={218}
                />
                <ModelMetricRow
                  model="Midjourney v6"
                  accuracy={94.3}
                  precision={93.7}
                  recall={94.8}
                  f1={94.2}
                  samples={187}
                />
                <ModelMetricRow
                  model="StyleGAN3"
                  accuracy={98.1}
                  precision={97.9}
                  recall={98.3}
                  f1={98.1}
                  samples={156}
                />
                <ModelMetricRow
                  model="Custom Models (3)"
                  accuracy={89.4}
                  precision={87.2}
                  recall={91.1}
                  f1={89.1}
                  samples={64}
                />
              </div>
            </TabsContent>

            <TabsContent value="calibration" className="space-y-4">
              <div className="p-6 bg-muted/40 rounded-lg space-y-4 border border-border">
                <div>
                  <h3 className="font-semibold mb-2">Calibration Error</h3>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold">3.2%</span>
                    <span className="text-sm text-emerald-400">Well calibrated</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Expected calibration error (ECE) measures alignment between confidence and accuracy
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="p-4 bg-background/40 rounded-lg border border-border">
                    <div className="text-sm text-muted-foreground mb-1">Temperature Scaling</div>
                    <div className="text-xl font-bold">1.23</div>
                  </div>
                  <div className="p-4 bg-background/40 rounded-lg border border-border">
                    <div className="text-sm text-muted-foreground mb-1">Brier Score</div>
                    <div className="text-xl font-bold">0.087</div>
                  </div>
                </div>

                <div className="mt-6">
                  <h3 className="font-semibold mb-3">Confidence Intervals</h3>
                  <div className="space-y-2">
                    <ConfidenceInterval model="Stable Diffusion XL" lower={88} upper={98} />
                    <ConfidenceInterval model="DALL-E 3" lower={85} upper={96} />
                    <ConfidenceInterval model="Midjourney v6" lower={83} upper={95} />
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* LOMO Cross-Validation Results */}
      <Card>
        <CardHeader>
          <CardTitle>LOMO Cross-Validation</CardTitle>
          <CardDescription>
            Leave-One-Model-Out testing to evaluate generalization to unseen models
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <LOMOResult model="Stable Diffusion XL" accuracy={94.7} confidence={91.2} />
            <LOMOResult model="DALL-E 3" accuracy={93.1} confidence={89.5} />
            <LOMOResult model="Midjourney v6" accuracy={92.8} confidence={88.9} />
            <LOMOResult model="StyleGAN3" accuracy={96.2} confidence={93.4} />
          </div>
          <p className="text-sm text-muted-foreground mt-4">
            LOMO validation shows strong generalization capability across different model families
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricCard({ 
  icon: Icon, 
  title, 
  value, 
  change, 
  positive, 
  target 
}: { 
  icon: any; 
  title: string; 
  value: string; 
  change: string; 
  positive: boolean;
  target: string;
}) {
  return (
    <div className="p-4 bg-muted/30 rounded-lg border border-border">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">{title}</span>
      </div>
      <div className="text-2xl font-bold mb-1">{value}</div>
      <div className="flex justify-between items-center">
        <span className={`text-xs ${positive ? 'text-green-600' : 'text-red-600'}`}>
          {change}
        </span>
        <span className="text-xs text-muted-foreground">Target: {target}</span>
      </div>
    </div>
  );
}

function ProgressBar({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-600',
    purple: 'bg-purple-600',
    green: 'bg-green-600',
    amber: 'bg-amber-600'
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{value}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div 
          className={`h-full ${colorClasses[color]} transition-all duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function ModelMetricRow({ 
  model, 
  accuracy, 
  precision, 
  recall, 
  f1, 
  samples 
}: { 
  model: string; 
  accuracy: number; 
  precision: number; 
  recall: number; 
  f1: number; 
  samples: number;
}) {
  return (
    <div className="p-4 bg-muted/30 rounded-lg border border-border">
      <div className="flex justify-between items-center mb-3">
        <span className="font-medium">{model}</span>
        <span className="text-sm text-muted-foreground">{samples} samples</span>
      </div>
      <div className="grid grid-cols-4 gap-4 text-sm">
        <div>
          <div className="text-muted-foreground mb-1">Accuracy</div>
          <div className="font-bold">{accuracy.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-muted-foreground mb-1">Precision</div>
          <div className="font-bold">{precision.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-muted-foreground mb-1">Recall</div>
          <div className="font-bold">{recall.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-muted-foreground mb-1">F1 Score</div>
          <div className="font-bold">{f1.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}

function ConfidenceInterval({ model, lower, upper }: { model: string; lower: number; upper: number }) {
  return (
    <div className="flex items-center gap-4">
      <span className="text-sm w-40 truncate">{model}</span>
      <div className="flex-1 relative h-8 bg-muted rounded">
        <div 
          className="absolute h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded"
          style={{ left: `${lower}%`, width: `${upper - lower}%` }}
        />
      </div>
      <span className="text-sm font-medium w-24 text-right">
        {lower}% - {upper}%
      </span>
    </div>
  );
}

function LOMOResult({ model, accuracy, confidence }: { model: string; accuracy: number; confidence: number }) {
  return (
    <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg border border-border">
      <span className="font-medium">{model}</span>
      <div className="flex gap-6 text-sm">
        <div>
          <span className="text-muted-foreground">Accuracy: </span>
          <span className="font-bold">{accuracy.toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-muted-foreground">Avg Confidence: </span>
          <span className="font-bold">{confidence.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
