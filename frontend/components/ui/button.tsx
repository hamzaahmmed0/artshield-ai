import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap font-sans text-sm font-semibold tracking-wide transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lab-accent focus-visible:ring-offset-2 focus-visible:ring-offset-lab-bg disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        primary:
          "border border-lab-accent/80 bg-lab-accent/10 text-lab-accent hover:bg-lab-accent/20 uppercase",
        ghost: "border border-lab-border bg-transparent text-neutral-300 hover:bg-lab-elevated hover:text-white",
        link: "border-0 text-lab-accent underline-offset-4 hover:underline",
      },
      size: {
        default: "h-12 px-8",
        lg: "h-14 px-10 text-base",
        sm: "h-9 px-4 text-xs",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  ),
);
Button.displayName = "Button";

export { Button, buttonVariants };
