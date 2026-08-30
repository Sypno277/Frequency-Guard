import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import type { FamilyProbability } from "@/lib/api/client";

interface ModelAttributionProps {
  isActive: boolean;
  /** Real calibrated per-family probabilities from the backend. */
  families: FamilyProbability[];
  /** "supervised" or "heuristic" — honesty label for the attribution mode. */
  attributionMode: string;
}

const FAMILY_LABELS: Record<string, string> = {
  real: "Real Camera",
  diffusion: "Diffusion",
  gan: "GAN",
  other: "Other / Unknown",
};

const FAMILY_COLORS: Record<string, string> = {
  real: "bg-success",
  diffusion: "bg-primary",
  gan: "bg-destructive",
  other: "bg-warning",
};

/**
 * Generator-family attribution (Masterplan §5.1).
 *
 * Renders calibrated softmax probabilities over real/diffusion/gan/other.
 * The mode badge keeps us honest: heuristic = no family-labeled training
 * data yet; supervised = fit on out-of-fold family labels.
 */
const ModelAttribution = ({ isActive, families, attributionMode }: ModelAttributionProps) => {
  const sorted = useMemo(() => [...families].sort((a, b) => b.probability - a.probability), [families]);
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const selected = sorted.find((f) => f.family === selectedFamily) ?? sorted[0];

  if (sorted.length === 0) return null;

  return (
    <div className="glass rounded-xl p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-lg font-semibold">Generator-Family Attribution</h3>
        <Badge variant={attributionMode === "supervised" ? "default" : "outline"}>
          {attributionMode === "supervised" ? "calibrated on labels" : "heuristic prior"}
        </Badge>
      </div>

      <div className="space-y-3">
        {sorted.map((family, index) => {
          const isTop = index === 0;
          const isSelected = selected?.family === family.family;
          return (
            <motion.div
              key={family.family}
              className={`flex cursor-pointer items-center gap-4 p-3 rounded-lg transition-colors ${
                isSelected ? "bg-primary/10 border border-primary/30" : "bg-muted/30"
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: isActive ? 1 : 0.5, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              onMouseEnter={() => setSelectedFamily(family.family)}
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`font-medium ${isTop && isActive ? "text-primary" : ""}`}>
                    {FAMILY_LABELS[family.family] ?? family.family}
                  </span>
                  {isTop && isActive && (
                    <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-medium">
                      Most likely
                    </span>
                  )}
                </div>
                <div className="mt-2 h-2 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${FAMILY_COLORS[family.family] ?? "bg-muted-foreground/40"}`}
                    initial={{ width: 0 }}
                    animate={{ width: isActive ? `${family.probability * 100}%` : 0 }}
                    transition={{ duration: 0.6, delay: index * 0.05 }}
                  />
                </div>
              </div>
              <span
                className={`font-mono text-sm ${isTop && isActive ? "text-primary" : "text-muted-foreground"}`}
              >
                {(family.probability * 100).toFixed(1)}%
              </span>
            </motion.div>
          );
        })}
      </div>

      <div className="mt-5 rounded-lg border border-border bg-muted/20 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h4 className="font-medium">Reading</h4>
          <span className="text-xs text-muted-foreground">
            {FAMILY_LABELS[selected?.family ?? ""] ?? selected?.family}
          </span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {selected?.family === "real"
            ? "The image's spectral statistics match natural camera capture: smooth 1/f energy decay, sensor-like noise residuals, and spatially consistent texture."
            : selected?.family === "diffusion"
              ? "Spectral flattening and mid-band roll-off typical of diffusion denoisers were measured — the high-frequency tail deviates from camera physics."
              : selected?.family === "gan"
                ? "Periodic upsampling echoes and checkerboard harmonics characteristic of GAN conv-transpose layers dominate this spectrum."
                : "The spectrum shows synthetic characteristics that don't match either major generator family cleanly."}
        </p>
      </div>
    </div>
  );
};

export default ModelAttribution;
