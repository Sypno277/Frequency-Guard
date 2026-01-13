import { Waves, Brain, Fingerprint, Gauge, Shield, Zap } from "lucide-react";
import { motion } from "framer-motion";

const Features = () => {
  const features = [
    {
      icon: Waves,
      title: "Frequency Domain Analysis",
      description: "Advanced FFT, DCT, and wavelet transforms to detect AI artifacts invisible to the human eye.",
    },
    {
      icon: Brain,
      title: "Multi-Branch Neural Network",
      description: "Parallel analysis of spatial, frequency, wavelet, and statistical features for maximum accuracy.",
    },
    {
      icon: Fingerprint,
      title: "Model Attribution",
      description: "Identify the specific AI model used: Stable Diffusion, Midjourney, DALL-E, or GAN variants.",
    },
    {
      icon: Gauge,
      title: "Real-Time Processing",
      description: "Get results in under 3 seconds with optimized inference and GPU acceleration.",
    },
    {
      icon: Shield,
      title: "Explainable AI",
      description: "Grad-CAM visualizations and SHAP values show exactly why an image was flagged.",
    },
    {
      icon: Zap,
      title: "Continuous Learning",
      description: "Adapts to new AI models through active learning and frequent model updates.",
    },
  ];

  return (
    <section id="features" className="py-20 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-muted/20 to-background" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
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
              className="glass rounded-2xl p-6 hover:bg-card/80 transition-all duration-300 group"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="h-6 w-6 text-primary" />
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
