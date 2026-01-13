import { ArrowRight, Scan, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroProps {
  onStartAnalysis: () => void;
}

const Hero = ({ onStartAnalysis }: HeroProps) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      {/* Gradient Mesh Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-10" />
        {/* Trust-inspiring gradient orbs */}
        <div className="orb-glow w-[500px] h-[500px] top-0 left-1/4 bg-gradient-to-br from-primary/20 to-primary/5" />
        <div className="orb-glow w-[600px] h-[600px] bottom-0 right-1/4 bg-gradient-to-br from-accent/15 to-accent/5" style={{ animationDelay: "-3s" }} />
        <div className="orb-glow w-[400px] h-[400px] top-1/2 left-0 bg-gradient-to-br from-primary/10 to-transparent" style={{ animationDelay: "-5s" }} />
        {/* Top radial gradient for trust feel */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,hsl(215_85%_55%/0.08)_0%,transparent_50%)]" />
      </div>
      
      {/* Floating Elements with gradient styling */}
      <div className="absolute top-32 left-20 hidden lg:block animate-float">
        <div className="gradient-card gradient-border rounded-xl p-4 glow-primary">
          <Scan className="h-6 w-6 text-primary" />
        </div>
      </div>
      <div className="absolute top-48 right-24 hidden lg:block animate-float" style={{ animationDelay: "1s" }}>
        <div className="gradient-card gradient-border rounded-xl p-4">
          <Shield className="h-6 w-6 text-success" />
        </div>
      </div>
      <div className="absolute bottom-32 left-32 hidden lg:block animate-float" style={{ animationDelay: "2s" }}>
        <div className="gradient-card gradient-border rounded-xl p-4">
          <Zap className="h-6 w-6 text-accent" />
        </div>
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge with gradient */}
          <div className="inline-flex items-center gap-2 gradient-card gradient-border rounded-full px-5 py-2.5 mb-8 fade-in">
            <span className="w-2 h-2 bg-gradient-to-r from-success to-accent rounded-full animate-pulse" />
            <span className="text-sm text-muted-foreground">Advanced AI Detection System</span>
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight fade-in">
            Detect AI-Generated
            <br />
            <span className="gradient-text glow-text">Images Instantly</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 fade-in" style={{ animationDelay: "0.2s" }}>
            State-of-the-art frequency domain analysis with multi-branch neural networks.
            Identify Stable Diffusion, Midjourney, DALL-E, and GAN-generated content.
          </p>

          {/* CTAs with enhanced gradients */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 fade-in" style={{ animationDelay: "0.4s" }}>
            <Button variant="hero" size="xl" onClick={onStartAnalysis} className="gradient-shine">
              Start Analysis
              <ArrowRight className="h-5 w-5" />
            </Button>
            <Button variant="glass" size="xl" className="gradient-border">
              View Documentation
            </Button>
          </div>

          {/* Stats with gradient cards */}
          <div className="grid grid-cols-3 gap-6 max-w-lg mx-auto mt-16 fade-in" style={{ animationDelay: "0.6s" }}>
            <div className="gradient-card gradient-border rounded-xl p-4">
              <div className="text-3xl font-bold gradient-text">95%+</div>
              <div className="text-sm text-muted-foreground">Accuracy</div>
            </div>
            <div className="gradient-card gradient-border rounded-xl p-4">
              <div className="text-3xl font-bold gradient-text">&lt;3s</div>
              <div className="text-sm text-muted-foreground">Analysis</div>
            </div>
            <div className="gradient-card gradient-border rounded-xl p-4">
              <div className="text-3xl font-bold gradient-text">6+</div>
              <div className="text-sm text-muted-foreground">AI Models</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-background via-background/80 to-transparent" />
    </section>
  );
};

export default Hero;
