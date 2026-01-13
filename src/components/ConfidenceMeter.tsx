import { motion } from "framer-motion";

interface ConfidenceMeterProps {
  value: number; // 0 to 100
  label: string;
  color: "real" | "ai" | "warning";
}

const ConfidenceMeter = ({ value, label, color }: ConfidenceMeterProps) => {
  const colorClasses = {
    real: "from-success to-success/50",
    ai: "from-destructive to-destructive/50",
    warning: "from-warning to-warning/50",
  };

  const bgClasses = {
    real: "bg-success/20",
    ai: "bg-destructive/20",
    warning: "bg-warning/20",
  };

  const textClasses = {
    real: "text-success",
    ai: "text-destructive",
    warning: "text-warning",
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className={`text-lg font-bold ${textClasses[color]}`}>{value}%</span>
      </div>
      <div className={`h-3 rounded-full overflow-hidden ${bgClasses[color]}`}>
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${colorClasses[color]}`}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};

export default ConfidenceMeter;
