import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-gradient-to-r from-primary to-primary/90 text-primary-foreground hover:from-primary/90 hover:to-primary shadow-lg hover:shadow-xl",
        destructive: "bg-gradient-to-r from-destructive to-destructive/90 text-destructive-foreground hover:from-destructive/90 hover:to-destructive",
        outline: "border border-border bg-background/50 hover:bg-accent/10 hover:text-accent-foreground hover:border-primary/50",
        secondary: "bg-gradient-to-r from-secondary to-secondary/90 text-secondary-foreground hover:from-secondary/80 hover:to-secondary/70",
        ghost: "hover:bg-primary/10 hover:text-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        // Hero variant - prominent CTA with trust gradient
        hero: "bg-gradient-to-r from-primary via-primary to-accent text-primary-foreground font-semibold shadow-lg hover:shadow-[0_0_40px_hsl(215,85%,55%,0.35)] hover:scale-105 transition-all duration-300",
        // Glow variant - with neon effect
        glow: "bg-gradient-to-r from-primary to-accent text-primary-foreground shadow-lg hover:shadow-[0_0_40px_hsl(215,85%,55%,0.4)] transition-all duration-300",
        // Glass variant - translucent with gradient tint
        glass: "bg-gradient-to-r from-card/80 to-card/60 backdrop-blur-md border border-border/50 text-foreground hover:from-card/90 hover:to-card/80 hover:border-primary/30",
        // Success variant
        success: "bg-gradient-to-r from-success to-success/90 text-success-foreground hover:from-success/90 hover:to-success",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-9 rounded-lg px-4",
        lg: "h-12 rounded-xl px-8 text-base",
        xl: "h-14 rounded-xl px-10 text-lg",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
