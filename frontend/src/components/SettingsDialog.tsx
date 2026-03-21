import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Monitor, Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function SettingsDialog({ open, onOpenChange }: Props) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Cài đặt</DialogTitle>
          <DialogDescription>Giao diện sáng / tối và hệ thống.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-2">
          <div className="space-y-3">
            <Label className="text-sm font-medium">Chế độ hiển thị</Label>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  { id: "light" as const, label: "Sáng", icon: Sun },
                  { id: "dark" as const, label: "Tối", icon: Moon },
                  { id: "system" as const, label: "Hệ thống", icon: Monitor },
                ] as const
              ).map(({ id, label, icon: Icon }) => (
                <Button
                  key={id}
                  type="button"
                  variant="outline"
                  className={cn(
                    "flex h-auto flex-col gap-1 py-3",
                    mounted && theme === id && "border-primary bg-sidebar-accent",
                  )}
                  onClick={() => setTheme(id)}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-xs">{label}</span>
                </Button>
              ))}
            </div>
            {mounted && (
              <p className="text-xs text-muted-foreground">
                Đang áp dụng:{" "}
                <span className="font-medium text-foreground">
                  {resolvedTheme === "dark" ? "Tối" : "Sáng"}
                </span>
                {theme === "system" ? " (theo hệ thống)" : ""}
              </p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
