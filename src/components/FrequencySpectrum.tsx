import { useMemo } from "react";
import { motion } from "framer-motion";

interface FrequencySpectrumProps {
  isActive: boolean;
}

const FrequencySpectrum = ({ isActive }: FrequencySpectrumProps) => {
  // Generate mock frequency data
  const frequencyData = useMemo(() => {
    return Array.from({ length: 64 }, (_, i) => {
      // Create a realistic-looking FFT magnitude pattern
      const centerBias = 1 - Math.abs(i - 32) / 32;
      const noise = Math.random() * 0.4;
      const aiArtifact = i > 20 && i < 44 ? Math.sin((i - 20) * 0.5) * 0.3 : 0;
      return Math.max(0.1, centerBias * 0.6 + noise + aiArtifact);
    });
  }, []);

  const getColor = (value: number, index: number) => {
    // Gradient from blue to cyan to green
    const hue = 180 + value * 60 + (index / 64) * 40;
    return `hsl(${hue}, 80%, ${40 + value * 30}%)`;
  };

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">FFT Magnitude Spectrum</h3>
      <div className="relative h-48 flex items-end justify-between gap-[2px] p-4 bg-muted/30 rounded-lg overflow-hidden">
        {/* Grid lines */}
        <div className="absolute inset-0 opacity-20">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="absolute left-0 right-0 border-t border-border"
              style={{ top: `${(i + 1) * 20}%` }}
            />
          ))}
        </div>

        {/* Frequency bars */}
        {frequencyData.map((value, index) => (
          <motion.div
            key={index}
            className="flex-1 rounded-t-sm"
            style={{
              background: isActive
                ? `linear-gradient(to top, ${getColor(value, index)}, ${getColor(value * 0.5, index)})`
                : "hsl(var(--muted-foreground) / 0.3)",
            }}
            initial={{ height: "10%" }}
            animate={{ height: isActive ? `${value * 100}%` : "10%" }}
            transition={{
              duration: 0.5,
              delay: index * 0.01,
              ease: "easeOut",
            }}
          />
        ))}

        {/* Overlay glow effect */}
        {isActive && (
          <div className="absolute inset-0 bg-gradient-to-t from-primary/10 to-transparent pointer-events-none" />
        )}
      </div>
      <div className="flex justify-between mt-2 text-xs text-muted-foreground">
        <span>0 Hz</span>
        <span>Nyquist</span>
      </div>
    </div>
  );
};

export default FrequencySpectrum;
