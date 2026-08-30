import { Layers } from "lucide-react";

interface HeatmapOverlayProps {
  /** Original image preview (data URL). When null, only the heatmap shows. */
  preview: string | null;
  /** Backend-generated saliency heatmap (base64 PNG data URL). */
  heatmapSrc: string;
  /** Overlay opacity 0..1. */
  opacity: number;
}

/**
 * Spectral-saliency overlay (Masterplan §5.1 explainability).
 *
 * Renders the backend's viridis heatmap on top of the uploaded image.
 * Yellow regions = tiles whose local FFT deviates most from the natural
 * 1/f power law — where synthetic content concentrates.
 */
const HeatmapOverlay = ({ preview, heatmapSrc, opacity }: HeatmapOverlayProps) => {
  return (
    <div className="relative rounded-xl overflow-hidden bg-muted/40 border border-border">
      {preview ? (
        <img src={preview} alt="Uploaded image" className="w-full max-h-[420px] object-contain" />
      ) : (
        <div className="flex items-center justify-center h-64 text-muted-foreground text-sm gap-2">
          <Layers className="h-5 w-5" />
          Saliency map (upload preview not retained for privacy)
        </div>
      )}
      <img
        src={heatmapSrc}
        alt="Spectral saliency heatmap"
        className="absolute inset-0 w-full h-full object-contain mix-blend-screen pointer-events-none transition-opacity duration-200"
        style={{ opacity }}
      />
      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-lg bg-background/80 backdrop-blur px-3 py-1.5 text-xs">
        <span>Natural</span>
        <div
          className="h-2 w-24 rounded"
          style={{
            background:
              "linear-gradient(to right, rgb(68,1,84), rgb(59,82,139), rgb(33,145,140), rgb(94,201,98), rgb(253,231,37))",
          }}
        />
        <span>1/f deviation ↑</span>
      </div>
    </div>
  );
};

export default HeatmapOverlay;
