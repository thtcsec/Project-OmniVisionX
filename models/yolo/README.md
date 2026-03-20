# YOLO weights (OmniVision)

Place Ultralytics YOLO weight files here (e.g. `yolo11m.pt`, `yolo11n.pt`). This directory is typically mounted into the `omni-object` container at `/app/weights` (and may be shared with other services).

- Do **not** commit `.pt` / `.onnx` / `.engine` binaries to git (ignored).
- You can download weights on first run via Ultralytics or copy them here for faster cold start.
