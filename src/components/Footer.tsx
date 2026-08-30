import { ShieldHalf, Github, Linkedin, Cpu } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="border-t-2 border-primary/20 py-12 bg-card/40">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <ShieldHalf className="h-6 w-6 text-primary" />
            <span className="font-bold gradient-text">Frequency Guard</span>
          </Link>

          {/* Links */}
          <div className="flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#" className="hover:text-primary transition-colors">Documentation</a>
            <Link to="/training" className="hover:text-primary transition-colors">Training</Link>
            <Link to="/dashboard" className="hover:text-primary transition-colors">Dashboard</Link>
            <a href="#" className="hover:text-primary transition-colors">Privacy</a>
          </div>

          {/* Social */}
          <div className="flex items-center gap-4">
            <a href="https://github.com/Sypno277/Frequency-Guard" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary transition-colors">
              <Github className="h-5 w-5" />
            </a>
            <a href="https://www.linkedin.com/in/pritish-borkar-a896a5318/" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary transition-colors">
              <Linkedin className="h-5 w-5" />
            </a>
          </div>
        </div>

        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground mt-8">
          <Cpu className="h-4 w-4" />
          <span>© 2026 Frequency Guard · CPU-only forensic engine · NLP narrative verdicts</span>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
