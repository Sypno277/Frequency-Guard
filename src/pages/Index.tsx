import { useState, useRef } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import HowItWorks from "@/components/HowItWorks";
import ImageUploader from "@/components/ImageUploader";
import AnalysisResults from "@/components/AnalysisResults";
import Footer from "@/components/Footer";

const Index = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<{
    isAI: boolean;
    confidence: number;
    model: string;
  } | null>(null);
  const [showResults, setShowResults] = useState(false);
  const uploaderRef = useRef<HTMLDivElement>(null);

  const handleStartAnalysis = () => {
    uploaderRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleImageUpload = async (file: File, preview: string) => {
    setIsAnalyzing(true);
    setShowResults(false);
    setAnalysisResult(null);

    // Simulate analysis delay (in production, this would call the backend API)
    await new Promise((resolve) => setTimeout(resolve, 3000));

    // Mock result - randomly determine if AI or real for demo
    const isAI = Math.random() > 0.4;
    const models = ["Stable Diffusion", "Midjourney", "DALL-E", "GAN (StyleGAN)", "Flux/SDXL"];
    const randomModel = models[Math.floor(Math.random() * models.length)];

    setAnalysisResult({
      isAI,
      confidence: isAI ? 75 + Math.random() * 20 : 80 + Math.random() * 18,
      model: isAI ? randomModel : "N/A",
    });

    setIsAnalyzing(false);
    setShowResults(true);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Hero onStartAnalysis={handleStartAnalysis} />
      <Features />
      <HowItWorks />
      
      <div ref={uploaderRef}>
        <ImageUploader onImageUpload={handleImageUpload} isAnalyzing={isAnalyzing} />
      </div>
      
      <AnalysisResults isVisible={showResults} result={analysisResult} />
      <Footer />
    </div>
  );
};

export default Index;
