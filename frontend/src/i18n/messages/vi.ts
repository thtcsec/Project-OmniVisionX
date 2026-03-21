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
    description: "Giao diện, ngôn ngữ và khóa tích hợp API (nếu bật).",
    tabGeneral: "Chung",
    tabIntegrations: "Tích hợp",
    integrationsTitle: "Tích hợp (sponsor)",
    integrationsHint: "API key / endpoint (file .env phía server).",
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
    save: "Lưu",
    saved: "Đã lưu",
    saveFailed: "Lưu thất bại",
    saving: "Đang lưu…",
    loading: "Đang tải…",
    envDisabled: "API đã tắt chỉnh sửa biến môi trường.",
    envFooter:
      "Thay đổi áp dụng cho tiến trình API. Container khác có thể cần restart để nhận giá trị mới.",
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
    troubleshootTitle: "Bị “Stream unavailable” hoặc ERR_CONNECTION_REFUSED cổng :8888 / :8889?",
    troubleshootP1:
      "HLS dùng cổng 8888; WebRTC (WHEP) dùng 8889 (mặc định MediaMTX). Không có service lắng nghe thì trình duyệt không tải được playlist hay WHEP.",
    troubleshootP2:
      "Tên path phải trùng ID camera. Lưu camera trạng thái online + URL RTSP sẽ tự đăng ký MediaMTX (API + Docker). Hoặc dùng relay omni-object khi bật detection.",
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
