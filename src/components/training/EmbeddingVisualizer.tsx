import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Layers, AlertCircle, CheckCircle, Eye } from 'lucide-react';
import type { ModelCluster, UnknownCluster } from '@/lib/types/training';

export default function EmbeddingVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [visualizationType, setVisualizationType] = useState<'tsne' | 'umap'>('tsne');
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);

  // Mock cluster data
  const clusters: ModelCluster[] = [
    {
      id: 'diffusion_sdxl',
      modelName: 'Stable Diffusion XL',
      modelFamily: 'Diffusion',
      centroid: [],
      variance: 0.23,
      sampleCount: 342,
      forensicFingerprint: { meanFeatures: {}, stdFeatures: {} },
      artifactPatterns: { commonArtifacts: [], artifactFrequency: {} },
      provenanceSignals: { commonEXIFPatterns: [], commonPromptKeywords: [], reverseImageSearchSources: [] },
      cohesion: 0.87,
      separation: 2.34,
      lastUpdated: Date.now()
    },
    {
      id: 'transformer_dalle3',
      modelName: 'DALL-E 3',
      modelFamily: 'Transformer',
      centroid: [],
      variance: 0.31,
      sampleCount: 218,
      forensicFingerprint: { meanFeatures: {}, stdFeatures: {} },
      artifactPatterns: { commonArtifacts: [], artifactFrequency: {} },
      provenanceSignals: { commonEXIFPatterns: [], commonPromptKeywords: [], reverseImageSearchSources: [] },
      cohesion: 0.82,
      separation: 2.89,
      lastUpdated: Date.now()
    },
    {
      id: 'gan_stylegan3',
      modelName: 'StyleGAN3',
      modelFamily: 'GAN',
      centroid: [],
      variance: 0.19,
      sampleCount: 156,
      forensicFingerprint: { meanFeatures: {}, stdFeatures: {} },
      artifactPatterns: { commonArtifacts: [], artifactFrequency: {} },
      provenanceSignals: { commonEXIFPatterns: [], commonPromptKeywords: [], reverseImageSearchSources: [] },
      cohesion: 0.91,
      separation: 3.12,
      lastUpdated: Date.now()
    }
  ];

  const unknownClusters: UnknownCluster[] = [
    {
      id: 'unknown_1',
      centroid: [],
      sampleCount: 23,
      averageCDCS: 0.76,
      representativeImages: [],
      forensicSignature: {},
      discoveredAt: Date.now() - 3 * 24 * 60 * 60 * 1000,
      status: 'pending_review'
    },
    {
      id: 'unknown_2',
      centroid: [],
      sampleCount: 17,
      averageCDCS: 0.81,
      representativeImages: [],
      forensicSignature: {},
      discoveredAt: Date.now() - 5 * 24 * 60 * 60 * 1000,
      status: 'pending_review'
    }
  ];

  useEffect(() => {
    renderEmbeddingVisualization();
  }, [visualizationType, selectedCluster]);

  const renderEmbeddingVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cssVar = (name: string, alpha = 1) => {
      const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return raw ? `hsl(${raw} / ${alpha})` : undefined;
    };

    // Clear canvas
    ctx.fillStyle = cssVar('--background') ?? '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw axes
    ctx.strokeStyle = cssVar('--border') ?? '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, canvas.height - 50);
    ctx.lineTo(canvas.width - 50, canvas.height - 50);
    ctx.moveTo(50, 50);
    ctx.lineTo(50, canvas.height - 50);
    ctx.stroke();

    // Add labels
    ctx.fillStyle = cssVar('--muted-foreground') ?? '#999';
    ctx.font = '12px sans-serif';
    ctx.fillText(visualizationType.toUpperCase() + ' Dimension 1', canvas.width / 2 - 50, canvas.height - 20);
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(visualizationType.toUpperCase() + ' Dimension 2', 0, 0);
    ctx.restore();

    // Draw cluster points
    const colors: Record<string, string> = {
      'Diffusion': '#3b82f6',
      'Transformer': '#8b5cf6',
      'GAN': '#10b981',
      'Unknown': '#f59e0b'
    };

    // Simulate cluster positions
    const clusterPositions: Record<string, { x: number; y: number }[]> = {};

    clusters.forEach((cluster, idx) => {
      const centerX = 150 + idx * 200;
      const centerY = 150 + idx * 100;
      const points: { x: number; y: number }[] = [];

      // Generate points around centroid
      for (let i = 0; i < cluster.sampleCount / 10; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 60 + 20;
        points.push({
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius
        });
      }

      clusterPositions[cluster.id] = points;

      // Draw points
      ctx.fillStyle = colors[cluster.modelFamily] || '#6b7280';
      ctx.globalAlpha = selectedCluster && selectedCluster !== cluster.id ? 0.2 : 0.6;
      
      points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw centroid
      ctx.fillStyle = colors[cluster.modelFamily] || '#6b7280';
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
      ctx.fill();

      // Draw label
      ctx.fillStyle = cssVar('--foreground') ?? '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(cluster.modelName, centerX + 15, centerY);
    });

    // Draw unknown clusters
    ctx.globalAlpha = 0.6;
    unknownClusters.forEach((cluster, idx) => {
      const x = 400 + idx * 150;
      const y = 400;

      ctx.fillStyle = colors['Unknown'];
      for (let i = 0; i < cluster.sampleCount / 3; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 40 + 10;
        ctx.beginPath();
        ctx.arc(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw unknown marker
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = cssVar('--foreground') ?? '#fff';
      ctx.font = '12px sans-serif';
      ctx.fillText(`Unknown #${idx + 1}`, x + 15, y);
    });

    ctx.globalAlpha = 1;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Layers className="w-5 h-5" />
                Embedding Space Visualization
              </CardTitle>
              <CardDescription>
                {visualizationType === 'tsne' ? 't-SNE' : 'UMAP'} projection of 512-dimensional embedding vectors
              </CardDescription>
            </div>
            <Select value={visualizationType} onValueChange={(v: 'tsne' | 'umap') => setVisualizationType(v)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="tsne">t-SNE</SelectItem>
                <SelectItem value="umap">UMAP</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <canvas
            ref={canvasRef}
            width={800}
            height={500}
            className="w-full border border-border rounded-lg bg-muted"
          />
        </CardContent>
      </Card>

      {/* Cluster Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Known Model Clusters</CardTitle>
            <CardDescription>Established clusters with high cohesion and separation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {clusters.map(cluster => (
              <ClusterCard
                key={cluster.id}
                cluster={cluster}
                isSelected={selectedCluster === cluster.id}
                onSelect={() => setSelectedCluster(cluster.id === selectedCluster ? null : cluster.id)}
              />
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-amber-600" />
              Unknown AI Clusters
            </CardTitle>
            <CardDescription>Pending review and labeling</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {unknownClusters.map((cluster, idx) => (
              <UnknownClusterCard key={cluster.id} cluster={cluster} index={idx + 1} />
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function ClusterCard({ 
  cluster, 
  isSelected, 
  onSelect 
}: { 
  cluster: ModelCluster; 
  isSelected: boolean; 
  onSelect: () => void;
}) {
  const familyColors: Record<string, string> = {
    'Diffusion': 'bg-primary/10 text-primary border border-primary/20',
    'Transformer': 'bg-accent/10 text-accent-foreground border border-accent/20',
    'GAN': 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
  };

  return (
    <div 
      className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
        isSelected ? 'border-primary bg-primary/5' : 'border-border hover:border-ring/40'
      }`}
      onClick={onSelect}
    >
      <div className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <div className="font-medium mb-1">{cluster.modelName}</div>
          <Badge className={familyColors[cluster.modelFamily]} variant="secondary">
            {cluster.modelFamily}
          </Badge>
        </div>
        {isSelected && <Eye className="w-4 h-4 text-primary" />}
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-muted-foreground">Samples</div>
          <div className="font-bold">{cluster.sampleCount}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Cohesion</div>
          <div className="font-bold">{(cluster.cohesion * 100).toFixed(0)}%</div>
        </div>
        <div>
          <div className="text-muted-foreground">Variance</div>
          <div className="font-bold">{cluster.variance.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Separation</div>
          <div className="font-bold">{cluster.separation.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}

function UnknownClusterCard({ cluster, index }: { cluster: UnknownCluster; index: number }) {
  return (
    <div className="p-4 border-2 border-amber-500/30 bg-amber-500/10 rounded-lg">
      <div className="flex justify-between items-start mb-3">
        <div>
          <div className="font-medium mb-1">Unknown Cluster #{index}</div>
          <div className="text-sm text-muted-foreground">
            Discovered {new Date(cluster.discoveredAt).toLocaleDateString()}
          </div>
        </div>
        <Badge variant="outline" className="border-amber-600 text-amber-600">
          Pending Review
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm mb-3">
        <div>
          <div className="text-muted-foreground">Samples</div>
          <div className="font-bold">{cluster.sampleCount}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Avg CDCS</div>
          <div className="font-bold">{(cluster.averageCDCS * 100).toFixed(0)}%</div>
        </div>
      </div>

      <Button variant="outline" size="sm" className="w-full">
        Review & Label
      </Button>
    </div>
  );
}
