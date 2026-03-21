import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Camera } from "@/types/omni";

export type CameraFormValues = {
  name: string;
  streamUrl: string;
  status: "online" | "offline";
};

const emptyForm: CameraFormValues = {
  name: "",
  streamUrl: "",
  status: "online",
};

function cameraToForm(cam: Camera): CameraFormValues {
  return {
    name: cam.name,
    streamUrl: cam.streamUrl ?? "",
    status: cam.status,
  };
}

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: "create" | "edit";
  camera?: Camera | null;
  onSubmit: (values: CameraFormValues) => Promise<void>;
  isSubmitting?: boolean;
};

export function CameraFormDialog({
  open,
  onOpenChange,
  mode,
  camera,
  onSubmit,
  isSubmitting,
}: Props) {
  const [form, setForm] = useState<CameraFormValues>(emptyForm);

  useEffect(() => {
    if (open) {
      setForm(mode === "edit" && camera ? cameraToForm(camera) : emptyForm);
    }
  }, [open, mode, camera]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(form);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>{mode === "create" ? "Add camera" : "Edit camera"}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="cam-name">Name</Label>
              <Input
                id="cam-name"
                value={form.name}
                onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                placeholder="e.g. Gate A"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="cam-stream">Stream URL (RTSP / HLS)</Label>
              <Input
                id="cam-stream"
                value={form.streamUrl}
                onChange={(e) => setForm((f) => ({ ...f, streamUrl: e.target.value }))}
                placeholder="rtsp://…"
              />
            </div>
            <div className="grid gap-2">
              <Label>Status</Label>
              <Select
                value={form.status}
                onValueChange={(v) =>
                  setForm((f) => ({ ...f, status: v as "online" | "offline" }))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="offline">Offline</SelectItem>
                  <SelectItem value="online">Online</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Saving…" : mode === "create" ? "Create" : "Save"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
