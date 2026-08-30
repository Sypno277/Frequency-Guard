import { ArrowRight, Scan, AudioWaveform, GitBranch, Cpu, Fingerprint } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroProps {
  onStartAnalysis: () => void;
}

/** Simulated spectrum bars for the hero visual (pure CSS, no data). */
const SpectrumVisual = () => {
  const bars = Array.from({ length: 48 }, (_, i) => i);
  return (
    <div className="relative h-24 flex items-end justify-between gap-[2px]">
      <div className="absolute inset-0 opacity-15">
        {[20, 40, 60, 80].map((top) => (
          <div key={top} className="absolute left-0 right-0 border-t border-border" style={{ top: `${top}%` }} />
        ))}
      </div>
      {bars.map((i) => {
        const height = 15 + Math.abs(Math.sin(i * 0.35)) * 75 + Math.abs(Math.cos(i * 0.11)) * 20;
        const hue = 189 + i * 1.5;
        return (
          <div
            key={i}
            className="flex-1 rounded-t-sm spectrum-bar border-t border-primary/30"
            style={{
              height: `${height}%`,
              background: `linear-gradient(to top, hsl(${hue} 94% 53% / 0.15), hsl(${hue} 94% 53%))`,
              animationDelay: `${i * 0.025}s`,
            }}
          />
        );
      })}
    </div>
  );
};

/**
 * Landing hero (Masterplan §5.1).
 *
 * Positions the product as a scientific forensic instrument: animated
 * frequency spectrum, a live "analysis pipeline" console, and metric
 * counters — with real CTA wiring into the uploader.
 */
const Hero = ({ onStartAnalysis }: HeroProps) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-20" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl" />

      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 mb-8 fade-in">
            <span className="w-2 h-2 bg-success rounded-full animate-pulse" />
            <span className="text-sm text-muted-foreground">Frequency-Domain Forensic Engine</span>
          </div>

          {/* Title */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight fade-in">
            Forensics-Grade
            <br />
            <span className="gradient-text">AI Image Detection</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 fade-in" style={{ animationDelay: "0.2s" }}>
            Multi-branch spectral + semantic analysis — FFT, DCT, wavelet, PRNU, and generator attribution —
            to separate authentic imagery from Stable Diffusion, Midjourney, DALL-E, and GAN artifacts.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 fade-in" style={{ animationDelay: "0.4s" }}>
            <Button variant="hero" size="xl" onClick={onStartAnalysis}>
              Start Analysis
              <ArrowRight className="h-5 w-5" />
            </Button>
            <Button
              variant="glass"
              size="xl"
              onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
            >
              View Documentation
            </Button>
          </div>

          {/* Live spectrum visual */}
          <div className="glass rounded-2xl p-6 mt-14 border-2 border-primary/20 fade-in" style={{ animationDelay: "0.6s" }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-sm font-medium">
                <AudioWaveform className="h-4 w-4 text-primary" />
                Live Frequency Spectrum
              </div>
              <span className="text-[11px] font-mono text-muted-foreground">
                FFT · DCT · Wavelet
              </span>
            </div>
            <SpectrumVisual />
          </div>

          {/* Pipeline console row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4 fade-in" style={{ animationDelay: "0.8s" }}>
            <div className="glass rounded-xl p-4 text-left">
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                <Scan className="h-3.5 w-3.5 text-primary" />
                Stage 01
              </div>
              <div className="font-mono text-sm">preprocess → fft</div>
            </div>
            <div className="glass rounded-xl p-4 text-left">
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                <GitBranch className="h-3.5 w-3.5 text-primary" />
                Stage 02
              </div>
              <div className="font-mono text-sm">feature_extract → classify</div>
            </div>
            <div className="glass rounded-xl p-4 text-left">
              <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                <Cpu className="h-3.5 w-3.5 text-primary" />
                Stage 03
              </div>
              <div className="font-mono text-sm">attribution → verdict</div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 max-w-lg mx-auto mt-12 fade-in" style={{ animationDelay: "1s" }}>
            <div>
              <div className="text-3xl font-bold gradient-text">6+</div>
              <div className="text-sm text-muted-foreground">Signal Branches</div>
            </div>
            <div>
              <div className="text-3xl font-bold gradient-text">{"<3s"}</div>
              <div className="text-sm text-muted-foreground">Analysis</div>
            </div>
            <div>
              <div className="text-3xl font-bold gradient-text flex items-center justify-center gap-1">
                <Fingerprint className="h-6 w-6" /> NLP
              </div>
              <div className="text-sm text-muted-foreground">Narrative Verdicts</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  );
};

export default Hero;
