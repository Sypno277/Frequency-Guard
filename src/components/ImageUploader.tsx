import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, X, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ImageUploaderProps {
  onImageUpload: (file: File, preview: string) => void;
  isAnalyzing: boolean;
  error?: string | null;
}

const ImageUploader = ({ onImageUpload, isAnalyzing, error }: ImageUploaderProps) => {
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
    <section id="demo" className="py-20">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Upload Your Image
            </h2>
            <p className="text-muted-foreground">
              Drag and drop an image or click to browse. Supports JPG, PNG, WebP, BMP.
            </p>
          </div>

          {error && (
            <div className="mb-6 flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div
            className={`
              relative rounded-2xl border-2 border-dashed transition-all duration-300
              ${isDragging 
                ? "border-primary bg-primary/5 scale-[1.02]" 
                : preview 
                  ? "border-border bg-card" 
                  : "border-border hover:border-primary/50 bg-card/50"
              }
              ${isAnalyzing ? "pulse-glow" : ""}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
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
                      <div className="absolute inset-0 bg-background/50 backdrop-blur-sm rounded-lg flex items-center justify-center">
                        <div className="flex flex-col items-center gap-3">
                          <Loader2 className="h-8 w-8 text-primary animate-spin" />
                          <span className="text-sm font-medium">Analyzing frequency patterns...</span>
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
                  <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
                    {isDragging ? (
                      <ImageIcon className="h-10 w-10 text-primary" />
                    ) : (
                      <Upload className="h-10 w-10 text-primary" />
                    )}
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
                    <span className="px-2 py-1 rounded bg-muted">JPG</span>
                    <span className="px-2 py-1 rounded bg-muted">PNG</span>
                    <span className="px-2 py-1 rounded bg-muted">WebP</span>
                    <span className="px-2 py-1 rounded bg-muted">BMP</span>
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
