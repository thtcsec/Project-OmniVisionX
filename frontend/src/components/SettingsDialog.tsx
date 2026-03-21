import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
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
import { fetchIntegrationEnvVars, updateIntegrationEnvVars, type IntegrationEnvVar } from "@/services/api";
import { toast } from "@/hooks/use-toast";

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export function SettingsDialog({ open, onOpenChange }: Props) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const { locale, setLocale, t } = useI18n();
  const [mounted, setMounted] = useState(false);
  const qc = useQueryClient();
  const [draft, setDraft] = useState<Record<string, string>>({});

  useEffect(() => {
    setMounted(true);
  }, []);

  const envQuery = useQuery({
    queryKey: ["integration-env"],
    queryFn: fetchIntegrationEnvVars,
    enabled: open,
    retry: false,
  });

  useEffect(() => {
    if (!open) return;
    if (!envQuery.data) return;
    const next: Record<string, string> = {};
    for (const v of envQuery.data) {
      next[v.key] = v.isSecret ? "" : (v.value ?? "");
    }
    setDraft(next);
  }, [open, envQuery.data]);

  const saveMut = useMutation({
    mutationFn: (updates: Array<{ key: string; value: string }>) => updateIntegrationEnvVars(updates),
    onSuccess: async () => {
      toast({ title: "Saved" });
      await qc.invalidateQueries({ queryKey: ["integration-env"] });
    },
    onError: (err) => toast({ title: "Save failed", description: err instanceof Error ? err.message : undefined, variant: "destructive" }),
  });

  const envVars = envQuery.data ?? [];
  const grouped = envVars.reduce<Record<string, IntegrationEnvVar[]>>((acc, v) => {
    (acc[v.group] ??= []).push(v);
    return acc;
  }, {});

  const envEnabled = !envQuery.isError;

  function onSaveEnv() {
    const updates = envVars.map((v) => ({
      key: v.key,
      value: (draft[v.key] ?? "").trim(),
    }));
    saveMut.mutate(updates);
  }

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

          <Separator />

          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <Label className="text-sm font-medium">Sponsor Integrations</Label>
              <Button type="button" size="sm" onClick={onSaveEnv} disabled={!envEnabled || envQuery.isLoading || saveMut.isPending}>
                Save
              </Button>
            </div>

            {!envEnabled && (
              <p className="text-xs text-muted-foreground">
                Env editor is disabled on the API.
              </p>
            )}

            {envEnabled && envQuery.isLoading && (
              <p className="text-xs text-muted-foreground">Loading…</p>
            )}

            {envEnabled && !envQuery.isLoading && (
              <div className="space-y-4">
                {Object.entries(grouped).map(([group, items]) => (
                  <div key={group} className="space-y-2">
                    <p className="text-xs font-semibold text-muted-foreground">{group}</p>
                    <div className="space-y-2">
                      {items.map((v) => {
                        const placeholder = v.isSecret
                          ? v.isSet ? "******** (set)" : "********"
                          : v.key;
                        return (
                          <div key={v.key} className="grid grid-cols-3 gap-2 items-center">
                            <Label className="text-xs col-span-1">{v.label}</Label>
                            <Input
                              className="col-span-2 h-9"
                              placeholder={placeholder}
                              value={draft[v.key] ?? ""}
                              onChange={(e) => setDraft((prev) => ({ ...prev, [v.key]: e.target.value }))}
                              type={v.isSecret ? "password" : "text"}
                              autoComplete="off"
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                <p className="text-xs text-muted-foreground">
                  Changes apply to the API process. Other containers may still need a restart to pick up new values.
                </p>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
