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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Eye, EyeOff, Monitor, Moon, Sun } from "lucide-react";
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
    setSecretVisible({});
  }, [open, envQuery.data]);

  const saveMut = useMutation({
    mutationFn: (updates: Array<{ key: string; value: string }>) => updateIntegrationEnvVars(updates),
    onSuccess: async () => {
      toast({ title: t("settings.saved") });
      await qc.invalidateQueries({ queryKey: ["integration-env"] });
    },
    onError: (err) =>
      toast({
        title: t("settings.saveFailed"),
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      }),
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
      <DialogContent className="flex max-h-[90vh] max-w-lg flex-col gap-0 overflow-hidden p-0 sm:max-w-lg">
        <DialogHeader className="shrink-0 space-y-1 border-b px-6 pb-4 pt-6 text-left">
          <DialogTitle>{t("settings.title")}</DialogTitle>
          <DialogDescription>{t("settings.description")}</DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="general" className="flex min-h-0 flex-1 flex-col px-6 pb-6 pt-2">
          <TabsList className="grid w-full shrink-0 grid-cols-2">
            <TabsTrigger value="general">{t("settings.tabGeneral")}</TabsTrigger>
            <TabsTrigger value="integrations">{t("settings.tabIntegrations")}</TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="mt-4 min-h-0 flex-1 space-y-6 overflow-y-auto pr-1 data-[state=inactive]:hidden">
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
          </TabsContent>

          <TabsContent
            value="integrations"
            className="mt-4 flex min-h-0 flex-1 flex-col gap-3 data-[state=inactive]:hidden"
          >
            <div className="flex shrink-0 flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-sm font-medium">{t("settings.integrationsTitle")}</p>
                <p className="text-xs text-muted-foreground">{t("settings.integrationsHint")}</p>
              </div>
              <Button
                type="button"
                size="sm"
                className="shrink-0 sm:mt-0"
                onClick={onSaveEnv}
                disabled={!envEnabled || envQuery.isLoading || saveMut.isPending}
              >
                {saveMut.isPending ? t("settings.saving") : t("settings.save")}
              </Button>
            </div>

            {!envEnabled && <p className="text-xs text-muted-foreground">{t("settings.envDisabled")}</p>}

            {envEnabled && envQuery.isLoading && (
              <p className="text-xs text-muted-foreground">{t("settings.loading")}</p>
            )}

            {envEnabled && !envQuery.isLoading && (
              <>
                <ScrollArea className="h-[min(52vh,380px)] rounded-md border bg-muted/20">
                  <div className="space-y-4 p-4 pr-3">
                    {Object.entries(grouped).map(([group, items]) => (
                      <div key={group} className="space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          {group}
                        </p>
                        <div className="space-y-3">
                          {items.map((v) => {
                            const placeholder = v.isSecret
                              ? v.isSet
                                ? "******** (set)"
                                : "********"
                              : v.key;
                            return (
                              <div key={v.key} className="grid gap-1.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] sm:items-center sm:gap-3">
                                <Label className="text-xs leading-snug sm:pt-0">{v.label}</Label>
                                {v.isSecret ? (
                                  <div className="relative">
                                    <Input
                                      className="h-9 pr-10"
                                      placeholder={placeholder}
                                      value={draft[v.key] ?? ""}
                                      onChange={(e) => setDraft((prev) => ({ ...prev, [v.key]: e.target.value }))}
                                      type={secretVisible[v.key] ? "text" : "password"}
                                      autoComplete="off"
                                      spellCheck={false}
                                    />
                                    <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon"
                                      className="absolute right-0 top-0 h-9 w-9 text-muted-foreground hover:text-foreground"
                                      onClick={() =>
                                        setSecretVisible((prev) => ({ ...prev, [v.key]: !prev[v.key] }))
                                      }
                                      aria-label={secretVisible[v.key] ? t("settings.hideSecret") : t("settings.showSecret")}
                                    >
                                      {secretVisible[v.key] ? (
                                        <EyeOff className="h-4 w-4" />
                                      ) : (
                                        <Eye className="h-4 w-4" />
                                      )}
                                    </Button>
                                  </div>
                                ) : (
                                  <Input
                                    className="h-9"
                                    placeholder={placeholder}
                                    value={draft[v.key] ?? ""}
                                    onChange={(e) => setDraft((prev) => ({ ...prev, [v.key]: e.target.value }))}
                                    type="text"
                                    autoComplete="off"
                                  />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                <p className="text-xs text-muted-foreground">{t("settings.envFooter")}</p>
              </>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
