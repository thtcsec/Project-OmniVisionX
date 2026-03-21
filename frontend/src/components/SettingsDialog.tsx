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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Monitor, Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n, type Locale } from "@/i18n/I18nProvider";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function SettingsDialog({ open, onOpenChange }: Props) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const { locale, setLocale, t } = useI18n();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("settings.title")}</DialogTitle>
          <DialogDescription>{t("settings.description")}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-6 py-2">
          <div className="space-y-2">
            <Label>{t("settings.language")}</Label>
            <Select value={locale} onValueChange={(v) => setLocale(v as Locale)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">{t("settings.english")}</SelectItem>
                <SelectItem value="vi">{t("settings.vietnamese")}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label className="text-sm font-medium">{t("settings.appearance")}</Label>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  { id: "light" as const, labelKey: "settings.light" as const, icon: Sun },
                  { id: "dark" as const, labelKey: "settings.dark" as const, icon: Moon },
                  { id: "system" as const, labelKey: "settings.system" as const, icon: Monitor },
                ] as const
              ).map(({ id, labelKey, icon: Icon }) => (
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
                  <span className="text-xs">{t(labelKey)}</span>
                </Button>
              ))}
            </div>
            {mounted && (
              <p className="text-xs text-muted-foreground">
                {t("settings.appliedPrefix")}{" "}
                <span className="font-medium text-foreground">
                  {resolvedTheme === "dark" ? t("settings.darkLabel") : t("settings.lightLabel")}
                </span>
                {theme === "system" ? ` ${t("settings.followSystem")}` : ""}
              </p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
