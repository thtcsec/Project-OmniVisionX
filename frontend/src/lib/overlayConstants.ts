/**
 * How long we keep last-known boxes on the UI without a new SignalR event.
 * omni-object may skip Redis publishes for stationary objects for up to
 * `stationary_publish_interval_s` (default 12s); 800ms/600ms was far too aggressive.
 */
export const OVERLAY_TRACK_TTL_MS = 35_000;
