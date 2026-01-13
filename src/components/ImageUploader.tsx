import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ImageUploaderProps {
  onImageUpload: (file: File, preview: string) => void;
  isAnalyzing: boolean;
}

const ImageUploader = ({ onImageUpload, isAnalyzing }: ImageUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      processFile(files[0]);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      processFile(files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      onImageUpload(file, result);
    };
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    setPreview(null);
  };

  return (
    <section id="demo" className="py-20 relative">
      {/* Background accent */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-br from-primary/5 to-accent/5 rounded-full blur-3xl" />
      </div>
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 gradient-card gradient-border rounded-full px-4 py-2 mb-6">
              <span className="text-sm text-muted-foreground">Quick Analysis</span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Upload Your <span className="gradient-text">Image</span>
            </h2>
            <p className="text-muted-foreground">
              Drag and drop an image or click to browse. Supports JPG, PNG, and WebP.
            </p>
          </div>

          <div
            className={`
              relative rounded-2xl border-2 border-dashed transition-all duration-300 gradient-card
              ${isDragging 
                ? "border-primary bg-primary/5 scale-[1.02] glow-primary" 
                : preview 
                  ? "border-border" 
                  : "border-border hover:border-primary/50"
              }
              ${isAnalyzing ? "pulse-glow" : ""}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {/* Gradient border effect */}
            <div className="absolute inset-0 rounded-2xl gradient-border pointer-events-none" />
            
            {/* Scan line effect when analyzing */}
            {isAnalyzing && (
              <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
                <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent scan-line" />
              </div>
            )}

            {preview ? (
              <div className="relative p-6">
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute top-4 right-4 z-10 bg-background/80 hover:bg-background"
                  onClick={clearImage}
                  disabled={isAnalyzing}
                >
                  <X className="h-4 w-4" />
                </Button>
                <div className="flex justify-center">
                  <div className="relative">
                    <img
                      src={preview}
                      alt="Uploaded preview"
                      className="max-h-[400px] rounded-lg object-contain"
                    />
                    {isAnalyzing && (
                      <div className="absolute inset-0 bg-gradient-to-br from-background/60 to-background/40 backdrop-blur-sm rounded-lg flex items-center justify-center">
                        <div className="flex flex-col items-center gap-3">
                          <div className="relative">
                            <Loader2 className="h-10 w-10 text-primary animate-spin" />
                            <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl" />
                          </div>
                          <span className="text-sm font-medium gradient-text">Analyzing frequency patterns...</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <label className="cursor-pointer block p-12">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
                <div className="flex flex-col items-center gap-6">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-primary to-accent rounded-2xl blur-xl opacity-30" />
                    <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center border border-primary/20">
                      {isDragging ? (
                        <ImageIcon className="h-10 w-10 text-primary" />
                      ) : (
                        <Upload className="h-10 w-10 text-primary" />
                      )}
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="font-medium mb-2">
                      {isDragging ? "Drop your image here" : "Drag & drop your image"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      or click to browse from your device
                    </p>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-muted to-muted/80 border border-border/50">JPG</span>
                    <span className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-muted to-muted/80 border border-border/50">PNG</span>
                    <span className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-muted to-muted/80 border border-border/50">WebP</span>
                  </div>
                </div>
              </label>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ImageUploader;
