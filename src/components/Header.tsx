import { ShieldHalf, Github, BookOpen, GraduationCap, BarChart3, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass border-b-2 border-primary/20">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <ShieldHalf className="h-8 w-8 text-primary transition-transform group-hover:scale-110" />
              <div className="absolute inset-0 blur-lg bg-primary/30" />
            </div>
            <span className="text-xl font-bold gradient-text">Frequency Guard</span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {[
              { label: "Features", href: "#features" },
              { label: "How It Works", href: "#how-it-works" },
              { label: "Demo", href: "#demo" },
            ].map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="text-muted-foreground hover:text-primary transition-colors font-medium"
              >
                {item.label}
              </a>
            ))}
            <Link to="/training" className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-1.5 font-medium">
              <GraduationCap className="h-4 w-4" />
              Training
            </Link>
            <Link to="/dashboard" className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-1.5 font-medium">
              <BarChart3 className="h-4 w-4" />
              Dashboard
            </Link>
          </nav>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <span className="hidden lg:flex items-center gap-1.5 text-xs text-success font-mono">
              <Activity className="h-3.5 w-3.5" />
              API ONLINE
            </span>
            <Button variant="ghost" size="icon" className="hidden sm:flex" asChild>
              <a href="https://github.com/Sypno277/Frequency-Guard" target="_blank" rel="noopener noreferrer">
                <Github className="h-5 w-5" />
              </a>
            </Button>
            <Button variant="ghost" size="icon" className="hidden sm:flex">
              <BookOpen className="h-5 w-5" />
            </Button>
            <Button variant="glow" size="sm" asChild>
              <Link to="/dashboard">Try Now</Link>
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
