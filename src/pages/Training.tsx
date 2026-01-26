import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Database, TrendingUp, Layers } from 'lucide-react';
import TrainingIntake from '@/components/training/TrainingIntake';
import TrainingDashboard from '@/components/training/TrainingDashboard';
import ModelManagement from '@/components/training/ModelManagement';
import EmbeddingVisualizer from '@/components/training/EmbeddingVisualizer';

export default function Training() {
  const [activeTab, setActiveTab] = useState('intake');

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

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Training Samples</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">1,247</div>
              <p className="text-xs text-muted-foreground">+89 this week</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Known Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">24</div>
              <p className="text-xs text-muted-foreground">+3 custom models</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">96.4%</div>
              <p className="text-xs text-emerald-400">+2.3% improvement</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Unknown Clusters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">7</div>
              <p className="text-xs text-amber-400">Pending review</p>
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
