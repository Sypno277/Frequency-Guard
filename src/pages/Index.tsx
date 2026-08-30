import { useState, useRef } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import HowItWorks from "@/components/HowItWorks";
import ImageUploader from "@/components/ImageUploader";
import AnalysisResults from "@/components/AnalysisResults";
import Footer from "@/components/Footer";
import { analyzeWithTrainingSignal, type AdaptiveAnalysisResult } from "@/lib/services/adaptiveInference";

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AdaptiveAnalysisResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const uploaderRef = useRef<HTMLDivElement>(null);

  const handleStartAnalysis = () => {
    uploaderRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Real pipeline: upload → POST /api/v1/analyze → render measured features.
  // No artificial delay; latency is whatever the backend reports.
  const handleImageUpload = async (file: File, preview: string) => {
    setIsAnalyzing(true);
    setShowResults(false);
    setAnalysisResult(null);
    setPreviewUrl(preview);
    setError(null);

    try {
      const result = await analyzeWithTrainingSignal(file);
      setAnalysisResult(result);
      setShowResults(true);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(
        err instanceof Error
          ? `${err.message} — is the API server running? (python -m uvicorn python_services.frequency_guard.api.server:app --port 8000)`
          : "Analysis failed unexpectedly."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Hero onStartAnalysis={handleStartAnalysis} />
      <Features />
      <HowItWorks />

      <div ref={uploaderRef}>
        <ImageUploader onImageUpload={handleImageUpload} isAnalyzing={isAnalyzing} error={error} />
      </div>

      <AnalysisResults isVisible={showResults} result={analysisResult} previewUrl={previewUrl} />
      <Footer />
    </div>
  );
};

export default Index;
