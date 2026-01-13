import { motion } from "framer-motion";

interface WaveletVisualizationProps {
  isActive: boolean;
}

const WaveletVisualization = ({ isActive }: WaveletVisualizationProps) => {
  const levels = [
    { name: "Level 1 (HH)", energy: 0.85, color: "from-destructive/80 to-destructive/40" },
    { name: "Level 2 (HL)", energy: 0.62, color: "from-warning/80 to-warning/40" },
    { name: "Level 3 (LH)", energy: 0.45, color: "from-primary/80 to-primary/40" },
    { name: "Level 4 (LL)", energy: 0.28, color: "from-accent/80 to-accent/40" },
  ];

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">Wavelet Decomposition</h3>
      <div className="space-y-4">
        {levels.map((level, index) => (
          <div key={level.name}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-muted-foreground">{level.name}</span>
              <span className="text-sm font-mono">
                {isActive ? (level.energy * 100).toFixed(1) : "0.0"}%
              </span>
            </div>
            <div className="h-6 bg-muted/30 rounded overflow-hidden">
              <motion.div
                className={`h-full rounded bg-gradient-to-r ${level.color}`}
                initial={{ width: 0 }}
                animate={{ width: isActive ? `${level.energy * 100}%` : 0 }}
                transition={{ duration: 0.8, delay: index * 0.1, ease: "easeOut" }}
              />
            </div>
          </div>
        ))}
      </div>
      
      {/* Decomposition matrix visualization */}
      <div className="mt-6 grid grid-cols-4 gap-2">
        {[...Array(16)].map((_, i) => (
          <motion.div
            key={i}
            className="aspect-square rounded"
            style={{
              background: isActive
                ? `hsl(${200 + Math.random() * 60}, 70%, ${30 + Math.random() * 30}%)`
                : "hsl(var(--muted))",
            }}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: isActive ? 1 : 0.3, scale: 1 }}
            transition={{ duration: 0.3, delay: i * 0.02 }}
          />
        ))}
      </div>
      <p className="text-xs text-muted-foreground mt-2 text-center">
        Haar wavelet 4-level decomposition
      </p>
    </div>
  );
};

export default WaveletVisualization;
