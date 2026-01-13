import { Shield, Github, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";

const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-strong">
      <div className="absolute inset-0 gradient-trust opacity-50" />
      <div className="container mx-auto px-4 relative">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-primary to-accent rounded-lg blur-md opacity-60" />
              <div className="relative bg-gradient-to-br from-primary to-accent rounded-lg p-1.5">
                <Shield className="h-5 w-5 text-primary-foreground" />
              </div>
            </div>
            <span className="text-xl font-bold gradient-text">DeepForensics</span>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-muted-foreground hover:text-foreground transition-colors relative group">
              Features
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-primary to-accent group-hover:w-full transition-all duration-300" />
            </a>
            <a href="#how-it-works" className="text-muted-foreground hover:text-foreground transition-colors relative group">
              How It Works
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-primary to-accent group-hover:w-full transition-all duration-300" />
            </a>
            <a href="#demo" className="text-muted-foreground hover:text-foreground transition-colors relative group">
              Demo
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-primary to-accent group-hover:w-full transition-all duration-300" />
            </a>
          </nav>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="hidden sm:flex hover:bg-primary/10">
              <Github className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="icon" className="hidden sm:flex hover:bg-primary/10">
              <BookOpen className="h-5 w-5" />
            </Button>
            <Button variant="hero" size="sm" className="gradient-border">
              Try Now
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
