import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Github, Sparkles } from "lucide-react";

const LOGO_SRC = "/branding/project-omnivisionx.png";

export default function About() {
  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div className="flex flex-col items-center text-center gap-4">
        <img
          src={LOGO_SRC}
          alt="OmniVisionX"
          className="h-28 w-28 md:h-32 md:w-32 object-contain rounded-2xl border bg-card p-2 shadow-sm"
        />
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">OmniVisionX</h1>
          <p className="mt-2 text-muted-foreground text-sm md:text-base max-w-md mx-auto">
            Nền tảng thị giác AI cho giao thông và hiểu cảnh — kiến trúc hướng sự kiện trên Redis Streams, API .NET 9 và
            cập nhật thời gian thực SignalR.
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Dự án</CardTitle>
          </div>
          <CardDescription>
            Phát triển trong khuôn khổ hackathon — demo nhanh, tập trung pipeline camera → phát hiện → biển số / người.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Separator />
          <div>
            <p className="text-sm font-medium text-muted-foreground uppercase tracking-wide">Developed by</p>
            <p className="text-xl font-semibold text-foreground mt-1">Trịnh Hoàng Tú</p>
          </div>
          <Separator />
          <a
            href="https://github.com/thtcsec/Project-OmniVisionX"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
          >
            <Github className="h-4 w-4" />
            Mã nguồn trên GitHub
          </a>
        </CardContent>
      </Card>
    </div>
  );
}
