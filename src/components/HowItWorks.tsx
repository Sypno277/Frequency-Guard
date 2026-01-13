import { motion } from "framer-motion";
import { Upload, Cpu, BarChart3, FileCheck } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      step: "01",
      title: "Upload Image",
      description: "Drag and drop any image file. We support JPG, PNG, and WebP formats up to 20MB.",
    },
    {
      icon: Cpu,
      step: "02",
      title: "Multi-Domain Analysis",
      description: "Our system extracts features from spatial, frequency, and wavelet domains simultaneously.",
    },
    {
      icon: BarChart3,
      step: "03",
      title: "Neural Network Inference",
      description: "A multi-branch CNN processes all features and computes detection probabilities.",
    },
    {
      icon: FileCheck,
      step: "04",
      title: "Detailed Report",
      description: "Get comprehensive results with visualizations, confidence scores, and model attribution.",
    },
  ];

  return (
    <section id="how-it-works" className="py-20">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            How <span className="gradient-text">It Works</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            A streamlined process powered by cutting-edge forensic algorithms
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="relative">
            {/* Connection line */}
            <div className="absolute left-8 top-12 bottom-12 w-px bg-gradient-to-b from-primary via-accent to-primary/20 hidden md:block" />

            <div className="space-y-12">
              {steps.map((step, index) => (
                <motion.div
                  key={step.step}
                  className="flex gap-6 items-start"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.15 }}
                  viewport={{ once: true }}
                >
                  <div className="relative flex-shrink-0">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                      <step.icon className="h-7 w-7 text-primary-foreground" />
                    </div>
                    <span className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-background border-2 border-primary text-xs font-bold flex items-center justify-center">
                      {index + 1}
                    </span>
                  </div>
                  <div className="glass rounded-xl p-6 flex-1">
                    <div className="text-xs font-mono text-primary mb-2">STEP {step.step}</div>
                    <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                    <p className="text-muted-foreground">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
