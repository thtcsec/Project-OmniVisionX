/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly OMNI_API_BASE_URL?: string;
  readonly OMNI_SIGNALR_URL?: string;
  readonly OMNI_SIGNALR_HUB_PATH?: string;
  readonly OMNI_SIMULATOR_BASE_URL?: string;
  readonly OMNI_MEDIA_BASE_URL?: string;
  /** WHEP base; default = same host as OMNI_MEDIA_BASE_URL with port 8889 */
  readonly OMNI_MEDIA_WEBRTC_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
