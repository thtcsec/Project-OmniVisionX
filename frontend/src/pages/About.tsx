import { Card, CardContent } from "@/components/ui/card";
import { Github, Sparkles } from "lucide-react";
import { useI18n } from "@/i18n/I18nProvider";
import { cn } from "@/lib/utils";

const LOGO_SRC = "/branding/project-omnivisionx.png";

export default function About() {
  const { t } = useI18n();

  return (
    <div className="mx-auto max-w-4xl">
      <Card className="overflow-hidden border-2 shadow-lg">
        <CardContent className="p-0">
          <div
            className={cn(
              "flex flex-col gap-8 p-6 sm:p-10 md:flex-row md:items-stretch md:gap-10",
              "bg-gradient-to-br from-primary/5 via-background to-muted/30",
            )}
          >
            {/* Logo — left, prominent */}
            <div className="flex shrink-0 flex-col items-center justify-center md:w-[min(42%,280px)]">
              <div
                className={cn(
                  "relative flex items-center justify-center rounded-3xl border-2 border-primary/20 bg-card p-6 shadow-xl",
                  "ring-4 ring-primary/10 dark:ring-primary/20",
                )}
              >
                <img
                  src={LOGO_SRC}
                  alt="OmniVisionX"
                  className="h-40 w-40 object-contain sm:h-48 sm:w-48 md:h-52 md:w-52"
                />
              </div>
              <p className="mt-4 text-center text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
                OmniVisionX
              </p>
            </div>

            {/* Copy — right */}
            <div className="flex min-w-0 flex-1 flex-col justify-center space-y-6">
              <div>
                <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">OmniVisionX</h1>
                <p className="mt-3 text-base leading-relaxed text-muted-foreground sm:text-lg">{t("about.tagline")}</p>
              </div>

              <div className="space-y-3 rounded-xl border bg-card/80 p-4 backdrop-blur-sm">
                <div className="flex items-center gap-2 text-foreground">
                  <Sparkles className="h-5 w-5 shrink-0 text-primary" />
                  <h2 className="text-lg font-semibold">{t("about.project")}</h2>
                </div>
                <p className="text-sm leading-relaxed text-muted-foreground">{t("about.projectDesc")}</p>
              </div>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {t("about.developedBy")}
                </p>
                <p className="mt-1 text-2xl font-semibold text-foreground">Trịnh Hoàng Tú</p>
              </div>

              <a
                href="https://github.com/thtcsec/Project-OmniVisionX"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex w-fit items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-2.5 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
              >
                <Github className="h-4 w-4" />
                {t("about.github")}
              </a>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
