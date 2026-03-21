import type { MessageTree } from "./en";

export const vi: MessageTree = {
  nav: {
    group: "Điều hướng",
    dashboard: "Bảng điều khiển",
    cameras: "Camera",
    plates: "Tra biển số",
    simulator: "Giả lập",
    live: "Xem trực tiếp",
    about: "Giới thiệu",
    settings: "Cài đặt",
    subtitle: "Phân tích giao thông",
  },
  settings: {
    title: "Cài đặt",
    description: "Giao diện và ngôn ngữ hiển thị.",
    appearance: "Chủ đề",
    light: "Sáng",
    dark: "Tối",
    system: "Hệ thống",
    appliedPrefix: "Đang dùng:",
    darkLabel: "Tối",
    lightLabel: "Sáng",
    followSystem: "(theo hệ thống)",
    language: "Ngôn ngữ",
    english: "English",
    vietnamese: "Tiếng Việt",
  },
  live: {
    title: "Xem trực tiếp",
    subtitle: "Luồng video và khung nhận diện realtime (SignalR).",
    selectCamera: "Camera",
    pickCamera: "Chọn camera",
    showTracks: "Hiện khung tracking",
    tracksHint:
      "Dữ liệu từ Redis → API (detections / vehicles / humans). Căn theo object-contain; giả định khung detector trùng độ phân giải stream.",
    noStream: "Chưa có URL HLS/WebRTC. Cấu hình trong Cameras hoặc MediaMTX.",
    waitingEvents: "Đang chờ sự kiện detection…",
    activeBoxes: "Khung đang hiện",
    joinSignalR: "Kết nối SignalR để nhận overlay.",
    streamUnavailable: "Không phát được luồng",
    noStreamConfigured: "Chưa cấu hình luồng",
  },
  about: {
    tagline:
      "Nền tảng thị giác AI cho giao thông và hiểu cảnh — kiến trúc hướng sự kiện Redis Streams, API .NET 9 và SignalR.",
    project: "Dự án",
    projectDesc:
      "Phát triển trong khuôn khổ hackathon — pipeline camera → phát hiện → biển số / người.",
    developedBy: "Developed by",
    github: "Mã nguồn trên GitHub",
  },
  cameras: {
    hackathonNote: "CRUD camera (hackathon — không đăng nhập). Production nên dùng JWT / API key.",
  },
};
