import { useState } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ServerInfoBanner from "@/components/dashboard/ServerInfoBanner";
import DashboardKpis from "@/components/dashboard/DashboardKpis";
import AnalyticsCharts from "@/components/dashboard/AnalyticsCharts";
import LiveMetrics from "@/components/dashboard/LiveMetrics";
import BatchAnalysis from "@/components/dashboard/BatchAnalysis";
import ModelPerformance from "@/components/dashboard/ModelPerformance";
import HistoryPanel from "@/components/dashboard/HistoryPanel";
import NarrativeFeed from "@/components/dashboard/NarrativeFeed";

/**
 * Data-science operations dashboard (Masterplan §5.2–§5.5).
 *
 * Layered analytics view: backend status, live KPI strip, derived
 * distributions, batch analysis, held-out model evaluation, and the
 * SQLite audit trail — all fed by the real FastAPI frequency-domain
 * backend.
 */
const Dashboard = () => {
  const [refreshKey, setRefreshKey] = useState(0);

  const bumpRefresh = () => setRefreshKey((k) => k + 1);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 pt-24 pb-12 space-y-6 max-w-7xl">
        {/* Page header */}
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold mb-1">
              Forensics <span className="gradient-text">Operations Center</span>
            </h1>
            <p className="text-muted-foreground text-sm">
              Live inference telemetry, detection distributions, held-out model evaluation, and the
              full audit trail — powered by the FastAPI frequency-domain backend.
            </p>
          </div>
          <div className="inline-flex items-center gap-2 glass rounded-full px-4 py-2 text-xs text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
            Live data · auto-refreshing
          </div>
        </div>

        {/* Backend / model status */}
        <ServerInfoBanner refreshKey={refreshKey} />

        {/* KPI strip */}
        <DashboardKpis refreshKey={refreshKey} />

        {/* Derived analytics */}
        <AnalyticsCharts refreshKey={refreshKey} />

        {/* Live service metrics */}
        <LiveMetrics refreshKey={refreshKey} />

        {/* Batch analysis */}
        <BatchAnalysis onJobComplete={bumpRefresh} />

        {/* FrequencyGuard Phase 5: linguistic narrative feed */}
        <NarrativeFeed refreshKey={refreshKey} />

        {/* Held-out model performance */}
        <ModelPerformance />

        {/* Audit history */}
        <HistoryPanel refreshKey={refreshKey} />
      </main>
      <Footer />
    </div>
  );
};

export default Dashboard;
