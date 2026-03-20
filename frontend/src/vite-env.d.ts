/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly OMNI_API_BASE_URL: string;
  readonly OMNI_SIGNALR_HUB_PATH: string;
  readonly OMNI_DEV_API_PROXY_TARGET: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
