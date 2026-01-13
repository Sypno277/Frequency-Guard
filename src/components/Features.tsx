import { Waves, Brain, Fingerprint, Gauge, Shield, Zap } from "lucide-react";
import { motion } from "framer-motion";

const Features = () => {
  const features = [
    {
      icon: Waves,
      title: "Frequency Domain Analysis",
      description: "Advanced FFT, DCT, and wavelet transforms to detect AI artifacts invisible to the human eye.",
      gradient: "from-primary to-accent",
    },
    {
      icon: Brain,
      title: "Multi-Branch Neural Network",
      description: "Parallel analysis of spatial, frequency, wavelet, and statistical features for maximum accuracy.",
      gradient: "from-accent to-success",
    },
    {
      icon: Fingerprint,
      title: "Model Attribution",
      description: "Identify the specific AI model used: Stable Diffusion, Midjourney, DALL-E, or GAN variants.",
      gradient: "from-primary to-primary",
    },
    {
      icon: Gauge,
      title: "Real-Time Processing",
      description: "Get results in under 3 seconds with optimized inference and GPU acceleration.",
      gradient: "from-success to-accent",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description: "Grad-CAM visualizations and SHAP values show exactly why an image was flagged.",
      gradient: "from-primary to-accent",
    },
    {
      icon: Zap,
      title: "Continuous Learning",
      description: "Adapts to new AI models through active learning and frequent model updates.",
      gradient: "from-accent to-primary",
    },
  ];

  return (
    <section id="features" className="py-24 relative overflow-hidden">
      {/* Gradient background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-gradient-to-b from-primary/10 to-transparent rounded-full blur-3xl" />
      </div>
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 gradient-card gradient-border rounded-full px-4 py-2 mb-6">
            <span className="text-sm text-muted-foreground">Trusted Technology</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Advanced Detection <span className="gradient-text">Technology</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            State-of-the-art forensic analysis combining signal processing with deep learning
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="gradient-card gradient-border rounded-2xl p-6 hover:scale-[1.02] transition-all duration-300 group gradient-shine"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-lg`}>
                <feature.icon className="h-6 w-6 text-primary-foreground" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
