import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Plus, Upload, Trash2, Edit, CheckCircle, AlertCircle } from 'lucide-react';
import type { CustomModel, ModelFamily } from '@/lib/types/training';

export default function ModelManagement() {
  const [models, setModels] = useState<CustomModel[]>([
    {
      id: '1',
      name: 'Stable Diffusion XL v1.0 Fine-tuned',
      family: 'Diffusion',
      description: 'Custom fine-tuned SDXL model on architectural images',
      exampleImages: [],
      clusterId: 'diffusion_sdxl_ft',
      createdBy: 'user123',
      createdAt: Date.now() - 7 * 24 * 60 * 60 * 1000,
      sampleCount: 47,
      averageConfidence: 0.923
    },
    {
      id: '2',
      name: 'Custom Portrait GAN',
      family: 'GAN',
      description: 'StyleGAN-based model specialized for portrait generation',
      exampleImages: [],
      clusterId: 'gan_portrait_custom',
      createdBy: 'user456',
      createdAt: Date.now() - 14 * 24 * 60 * 60 * 1000,
      sampleCount: 32,
      averageConfidence: 0.891
    },
    {
      id: '3',
      name: 'DALL-E 3 Variant',
      family: 'Transformer',
      description: 'Experimental DALL-E variant with enhanced prompt following',
      exampleImages: [],
      clusterId: 'transformer_dalle3_var',
      createdBy: 'user789',
      createdAt: Date.now() - 21 * 24 * 60 * 60 * 1000,
      sampleCount: 28,
      averageConfidence: 0.876
    }
  ]);

  const [isAddModelOpen, setIsAddModelOpen] = useState(false);

  return (
    <div className="space-y-6">
      {/* Header with Add Button */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Custom Model Management</h2>
          <p className="text-muted-foreground">
            Add and manage custom AI models with few-shot learning (5-10 examples)
          </p>
        </div>
        <Dialog open={isAddModelOpen} onOpenChange={setIsAddModelOpen}>
          <DialogTrigger asChild>
            <Button className="flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Add Custom Model
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <AddModelDialog onClose={() => setIsAddModelOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map(model => (
          <ModelCard key={model.id} model={model} />
        ))}
      </div>

      {/* Known Models Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Known Model Families</CardTitle>
          <CardDescription>
            Pre-configured models with established clusters and forensic fingerprints
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <KnownModelCard
              family="Diffusion"
              models={['Stable Diffusion XL', 'Stable Diffusion v2.1', 'Stable Diffusion v1.5']}
              samples={687}
              accuracy={96.8}
            />
            <KnownModelCard
              family="GAN"
              models={['StyleGAN3', 'StyleGAN2']}
              samples={412}
              accuracy={97.3}
            />
            <KnownModelCard
              family="Transformer"
              models={['DALL-E 3', 'DALL-E 2']}
              samples={523}
              accuracy={95.4}
            />
            <KnownModelCard
              family="Real Images"
              models={['Natural Photos', 'Camera Captures']}
              samples={892}
              accuracy={98.7}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ModelCard({ model }: { model: CustomModel }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <CardTitle className="text-lg">{model.name}</CardTitle>
            <Badge variant="secondary" className="mt-2">
              {model.family}
            </Badge>
          </div>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost">
              <Edit className="w-4 h-4" />
            </Button>
            <Button size="sm" variant="ghost">
              <Trash2 className="w-4 h-4 text-red-600" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{model.description}</p>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="p-3 bg-primary/5 rounded-lg border border-primary/20">
            <div className="text-muted-foreground mb-1">Samples</div>
            <div className="font-bold">{model.sampleCount}</div>
          </div>
          <div className="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <div className="text-muted-foreground mb-1">Confidence</div>
            <div className="font-bold">{(model.averageConfidence * 100).toFixed(1)}%</div>
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          Created {new Date(model.createdAt).toLocaleDateString()}
        </div>

        <Button variant="outline" className="w-full">
          View Cluster Details
        </Button>
      </CardContent>
    </Card>
  );
}

function KnownModelCard({ 
  family, 
  models, 
  samples, 
  accuracy 
}: { 
  family: string; 
  models: string[]; 
  samples: number; 
  accuracy: number;
}) {
  return (
    <div className="p-4 bg-gradient-to-br from-primary/10 to-accent/10 rounded-lg border border-border">
      <div className="flex items-center gap-2 mb-3">
        <CheckCircle className="w-5 h-5 text-green-600" />
        <h3 className="font-semibold">{family}</h3>
      </div>
      
      <div className="space-y-2 mb-4">
        {models.map(model => (
          <div key={model} className="text-sm text-muted-foreground pl-4">
            • {model}
          </div>
        ))}
      </div>

      <div className="flex justify-between text-sm pt-3 border-t border-border">
        <span className="text-muted-foreground">{samples} samples</span>
        <span className="font-medium text-emerald-400">{accuracy}% accuracy</span>
      </div>
    </div>
  );
}

function AddModelDialog({ onClose }: { onClose: () => void }) {
  const [step, setStep] = useState<'details' | 'examples' | 'review'>('details');
  const [modelName, setModelName] = useState('');
  const [modelFamily, setModelFamily] = useState<ModelFamily>('Diffusion');
  const [description, setDescription] = useState('');
  const [exampleFiles, setExampleFiles] = useState<File[]>([]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setExampleFiles([...exampleFiles, ...files].slice(0, 10));
  };

  const handleSubmit = () => {
    // Create model logic here
    console.log('Creating model:', { modelName, modelFamily, description, exampleCount: exampleFiles.length });
    onClose();
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>Add Custom Model</DialogTitle>
        <DialogDescription>
          Few-shot learning: Provide 5-10 example images to create a new model cluster
        </DialogDescription>
      </DialogHeader>

      {step === 'details' && (
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>Model Name</Label>
            <Input
              placeholder="e.g., My Custom Stable Diffusion XL v2.0"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Model Family</Label>
            <Select value={modelFamily} onValueChange={(v) => setModelFamily(v as ModelFamily)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Diffusion">Diffusion</SelectItem>
                <SelectItem value="GAN">GAN</SelectItem>
                <SelectItem value="Transformer">Transformer</SelectItem>
                <SelectItem value="Unknown">Unknown</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Description</Label>
            <Textarea
              placeholder="Describe the model's purpose and characteristics..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>

          <Button 
            onClick={() => setStep('examples')} 
            className="w-full"
            disabled={!modelName.trim() || !description.trim()}
          >
            Next: Upload Examples
          </Button>
        </div>
      )}

      {step === 'examples' && (
        <div className="space-y-4 py-4">
          <div className="p-6 border-2 border-dashed border-border rounded-lg text-center">
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-sm font-medium mb-2">Upload 5-10 Example Images</p>
            <p className="text-xs text-muted-foreground mb-4">
              These examples will be used to generate the model's cluster centroid
            </p>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              id="example-upload"
            />
            <label htmlFor="example-upload">
              <Button variant="outline" asChild>
                <span>Choose Files</span>
              </Button>
            </label>
          </div>

          {exampleFiles.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium">
                Uploaded Examples: {exampleFiles.length}/10
              </p>
              <div className="grid grid-cols-3 gap-2">
                {exampleFiles.map((file, idx) => (
                  <div key={idx} className="relative aspect-square bg-muted rounded-lg overflow-hidden border border-border">
                    <img
                      src={URL.createObjectURL(file)}
                      alt={`Example ${idx + 1}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {exampleFiles.length >= 5 && exampleFiles.length <= 10 && (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-950 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-sm">Ready to create model cluster</span>
            </div>
          )}

          {exampleFiles.length < 5 && (
            <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950 rounded-lg">
              <AlertCircle className="w-5 h-5 text-amber-600" />
              <span className="text-sm">Need at least 5 examples for few-shot learning</span>
            </div>
          )}

          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setStep('details')} className="flex-1">
              Back
            </Button>
            <Button 
              onClick={() => setStep('review')} 
              className="flex-1"
              disabled={exampleFiles.length < 5 || exampleFiles.length > 10}
            >
              Next: Review
            </Button>
          </div>
        </div>
      )}

      {step === 'review' && (
        <div className="space-y-4 py-4">
          <div className="p-4 bg-muted/40 rounded-lg space-y-3 border border-border">
            <div>
              <div className="text-sm text-muted-foreground">Model Name</div>
              <div className="font-medium">{modelName}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Family</div>
              <Badge>{modelFamily}</Badge>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Description</div>
              <div className="text-sm">{description}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Example Images</div>
              <div className="font-medium">{exampleFiles.length} images</div>
            </div>
          </div>

          <div className="p-4 bg-primary/5 rounded-lg space-y-2 border border-primary/20">
            <h4 className="font-semibold text-sm">What happens next?</h4>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>• Extract embeddings from example images</li>
              <li>• Compute cluster centroid and variance</li>
              <li>• Generate statistical fingerprint distribution</li>
              <li>• Fine-tune embedding backbone (MAML)</li>
              <li>• Update detector with new model</li>
            </ul>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setStep('examples')} className="flex-1">
              Back
            </Button>
            <Button onClick={handleSubmit} className="flex-1">
              Create Model
            </Button>
          </div>
        </div>
      )}
    </>
  );
}
