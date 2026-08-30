import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import type { SpectrumBin, AzimuthalPoint } from "@/lib/api/client";

interface FrequencySpectrumProps {
  isActive: boolean;
  /** Real radial spectral bins from the backend pipeline. */
  spectrum: SpectrumBin[];
  /** Real azimuthal power profile from the backend pipeline. */
  azimuthal: AzimuthalPoint[];
}

/**
 * FFT Spectrum Explorer (Masterplan §5.1) — renders measured data only.
 *
 * The radial profile is the log-spaced mean magnitude per annulus. The
 * azimuthal profile is the angular energy distribution, which exposes
 * directional / checkerboard artifacts invisible to radial-only analysis.
 * No synthetic PRNG: every bar is a value computed from the image.
 */
const FrequencySpectrum = ({ isActive, spectrum, azimuthal }: FrequencySpectrumProps) => {
  const [metric, setMetric] = useState<"magnitude" | "energy">("magnitude");
  const [hoveredBin, setHoveredBin] = useState<number | null>(null);

  // Merge the two real profiles into one chart series.
  // Radial bins are labeled by normalized frequency; azimuthal bins by angle.
  const series = useMemo(() => {
    const radial = spectrum.map((p) => ({
      key: `radial-${p.bin}`,
      label: p.bin,
      metricLabel: (p.frequency * 100).toFixed(0),
      value: p.magnitude,
      kind: "radial" as const,
    }));
    const az = azimuthal.map((p) => ({
      key: `az-${p.angle_deg}`,
      label: p.angle_deg,
      metricLabel: `${p.angle_deg}°`,
      value: p.magnitude,
      kind: "azimuthal" as const,
    }));
    return [...radial, ...az];
  }, [spectrum, azimuthal]);

  const averageMagnitude = useMemo(
    () => (series.length ? series.reduce((sum, p) => sum + p.value, 0) / series.length : 0),
    [series]
  );

  // For the "energy" toggle we approximate power as magnitude^2 (matches backend scaling).
  const averageEnergy = useMemo(
    () => (series.length ? series.reduce((sum, p) => sum + p.value * p.value, 0) / series.length : 0),
    [series]
  );

  const getColor = (value: number, index: number) => {
    const hue = 180 + value * 60 + (index / Math.max(1, series.length)) * 40;
    return `hsl(${hue}, 80%, ${40 + value * 30}%)`;
  };

  const selectedPoint =
    hoveredBin !== null
      ? series[hoveredBin]
      : series[Math.floor(series.length / 2)];

  if (!series.length) {
    return (
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-2">FFT Spectrum Explorer</h3>
        <p className="text-sm text-muted-foreground">No spectral data available.</p>
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold">FFT Spectrum Explorer</h3>
        <div className="flex items-center gap-2">
          <Button
            variant={metric === "magnitude" ? "default" : "secondary"}
            size="sm"
            onClick={() => setMetric("magnitude")}
          >
            Magnitude
          </Button>
          <Button
            variant={metric === "energy" ? "default" : "secondary"}
            size="sm"
            onClick={() => setMetric("energy")}
          >
            Energy
          </Button>
        </div>
      </div>

      <div className="relative h-48 flex items-end justify-between gap-[2px] p-4 bg-muted/30 rounded-lg overflow-hidden">
        <div className="absolute inset-0 opacity-20">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="absolute left-0 right-0 border-t border-border"
              style={{ top: `${(i + 1) * 20}%` }}
            />
          ))}
        </div>

        {series.map((point, index) => {
          const plottedValue = metric === "magnitude" ? point.value : point.value * point.value;
          return (
            <motion.div
              key={point.key}
              className={`flex-1 rounded-t-sm transition-all ${hoveredBin === index ? "opacity-100" : "opacity-90"}`}
              style={{
                background: isActive
                  ? `linear-gradient(to top, ${getColor(plottedValue, index)}, ${getColor(plottedValue * 0.5, index)})`
                  : "hsl(var(--muted-foreground) / 0.3)",
              }}
              initial={{ height: "10%" }}
              animate={{ height: isActive ? `${Math.min(100, plottedValue * 100)}%` : "10%" }}
              transition={{ duration: 0.5, delay: index * 0.01, ease: "easeOut" }}
              onMouseEnter={() => setHoveredBin(index)}
              onMouseLeave={() => setHoveredBin(null)}
              title={`${point.kind === "radial" ? `Radial bin ${point.label}` : `Angle ${point.label}°`}: ${plottedValue.toFixed(4)}`}
            />
          );
        })}

        {isActive && (
          <div className="absolute inset-0 bg-gradient-to-t from-primary/10 to-transparent pointer-events-none" />
        )}
      </div>

      <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="rounded-lg bg-muted/30 px-3 py-2">
          <div className="text-muted-foreground">Selected {selectedPoint.kind}</div>
          <div className="font-mono text-sm">
            {selectedPoint.kind === "radial"
              ? `${selectedPoint.label} (${selectedPoint.metricLabel}%)`
              : `${selectedPoint.label}°`}
          </div>
        </div>
        <div className="rounded-lg bg-muted/30 px-3 py-2">
          <div className="text-muted-foreground">Magnitude Avg</div>
          <div className="font-mono text-sm">{averageMagnitude.toFixed(3)}</div>
        </div>
        <div className="rounded-lg bg-muted/30 px-3 py-2">
          <div className="text-muted-foreground">Energy Avg</div>
          <div className="font-mono text-sm">{averageEnergy.toFixed(3)}</div>
        </div>
      </div>

      <div className="flex justify-between mt-2 text-xs text-muted-foreground">
        <span>Radial bins</span>
        <span>Azimuthal profile →</span>
      </div>
    </div>
  );
};

export default FrequencySpectrum;
