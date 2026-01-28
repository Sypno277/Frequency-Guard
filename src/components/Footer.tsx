import { Shield, Github, Linkedin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t border-border py-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <Shield className="h-6 w-6 text-primary" />
            <span className="font-bold gradient-text">Frequency Guard</span>
          </div>

          {/* Links */}
          <div className="flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#" className="hover:text-foreground transition-colors">Documentation</a>
            <a href="#" className="hover:text-foreground transition-colors">API</a>
            <a href="#" className="hover:text-foreground transition-colors">Research</a>
            <a href="#" className="hover:text-foreground transition-colors">Privacy</a>
          </div>

          {/* Social */}
          <div className="flex items-center gap-4">
            <a href="https://github.com/Sypno277/Frequency-Guard" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
              <Github className="h-5 w-5" />
            </a>
            <a href="https://www.linkedin.com/in/pritish-borkar-a896a5318/" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
              <Linkedin className="h-5 w-5" />
            </a>
          </div>
        </div>

        <div className="text-center text-sm text-muted-foreground mt-8">
          © 2024 Frequency Guard. Advanced Image Authenticity Detection.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
