import { motion } from "framer-motion";

interface ModelAttributionProps {
  isActive: boolean;
  detectedModel: string | null;
}

const ModelAttribution = ({ isActive, detectedModel }: ModelAttributionProps) => {
  const models = [
    { name: "Stable Diffusion", probability: detectedModel === "Stable Diffusion" ? 0.87 : 0.05 },
    { name: "Midjourney", probability: detectedModel === "Midjourney" ? 0.82 : 0.08 },
    { name: "DALL-E", probability: detectedModel === "DALL-E" ? 0.79 : 0.03 },
    { name: "GAN (StyleGAN)", probability: detectedModel === "GAN" ? 0.75 : 0.02 },
    { name: "Flux/SDXL", probability: detectedModel === "Flux" ? 0.71 : 0.01 },
    { name: "Real Image", probability: detectedModel === "Real" ? 0.95 : 0.01 },
  ];

  const sortedModels = [...models].sort((a, b) => b.probability - a.probability);

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">Model Attribution</h3>
      <div className="space-y-3">
        {sortedModels.map((model, index) => {
          const isDetected = model.probability > 0.5;
          return (
            <motion.div
              key={model.name}
              className={`flex items-center gap-4 p-3 rounded-lg transition-colors ${
                isDetected && isActive ? "bg-primary/10 border border-primary/30" : "bg-muted/30"
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: isActive ? 1 : 0.5, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`font-medium ${isDetected && isActive ? "text-primary" : ""}`}>
                    {model.name}
                  </span>
                  {isDetected && isActive && (
                    <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-medium">
                      Detected
                    </span>
                  )}
                </div>
                <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      isDetected ? "bg-primary" : "bg-muted-foreground/30"
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: isActive ? `${model.probability * 100}%` : 0 }}
                    transition={{ duration: 0.6, delay: index * 0.05 }}
                  />
                </div>
              </div>
              <span className={`font-mono text-sm ${isDetected && isActive ? "text-primary" : "text-muted-foreground"}`}>
                {isActive ? `${(model.probability * 100).toFixed(1)}%` : "—"}
              </span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelAttribution;
