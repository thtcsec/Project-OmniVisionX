using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace Omni.API.Services;

/// <summary>
/// Registers camera RTSP sources as MediaMTX paths so HLS/WebRTC URLs
/// <c>/{cameraId}/index.m3u8</c> and <c>/{cameraId}/whep</c> resolve.
/// </summary>
public sealed class MediaMtxPathRegistrar
{
    private readonly HttpClient _http;
    private readonly IConfiguration _configuration;
    private readonly ILogger<MediaMtxPathRegistrar> _logger;

    public MediaMtxPathRegistrar(
        HttpClient http,
        IConfiguration configuration,
        ILogger<MediaMtxPathRegistrar> logger)
    {
        _http = http;
        _configuration = configuration;
        _logger = logger;
    }

    public bool IsConfigured =>
        !string.IsNullOrWhiteSpace(_configuration["MediaMtx:ApiBaseUrl"]);

    private string? ApiBaseUrl => _configuration["MediaMtx:ApiBaseUrl"]?.TrimEnd('/');
    private bool H264RelayEnabled => string.Equals(_configuration["MediaMtx:H264RelayEnabled"], "true", StringComparison.OrdinalIgnoreCase);
    private string SimulatorApiBaseUrl => (_configuration["Simulator:ApiBaseUrl"] ?? "http://omni-simulator:8554").TrimEnd('/');

    /// <returns>true if skipped (not RTSP / offline / no config) or HTTP 200/204.</returns>
    public async Task<bool> TryRegisterPathAsync(string cameraId, string streamUrl, CancellationToken ct = default)
    {
        var baseUrl = ApiBaseUrl;
        if (string.IsNullOrEmpty(baseUrl))
            return true;

        if (string.IsNullOrWhiteSpace(streamUrl)
            || !streamUrl.Trim().StartsWith("rtsp", StringComparison.OrdinalIgnoreCase))
            return true;

        if (IsMediaMtxSelfSource(streamUrl))
            return true;

        var path = Uri.EscapeDataString(cameraId);
        var url = $"{baseUrl}/v3/config/paths/replace/{path}";
        var payload = new MediaMtxPathPayload();

        if (H264RelayEnabled)
        {
            var relayOk = await TryStartSimulatorRelayAsync(cameraId, streamUrl.Trim(), ct);
            if (!relayOk)
                return false;

            payload.Source = "publisher";
            payload.SourceOnDemand = false;
        }
        else
        {
            payload.Source = streamUrl.Trim();
            payload.SourceOnDemand = true;
            payload.SourceOnDemandStartTimeout = "15s";
            payload.SourceOnDemandCloseAfter = "5m";
        }

        try
        {
            var res = await _http.PostAsJsonAsync(url, payload, ct);
            if (res.IsSuccessStatusCode)
            {
                _logger.LogDebug("MediaMTX path {CameraId} registered", cameraId);
                return true;
            }

            var body = await res.Content.ReadAsStringAsync(ct);
            _logger.LogWarning(
                "MediaMTX path register failed for {CameraId}: {Status} {Body}",
                cameraId,
                (int)res.StatusCode,
                body.Length > 200 ? body[..200] : body);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "MediaMTX API unreachable for camera {CameraId}", cameraId);
            return false;
        }
    }

    private async Task<bool> TryStartSimulatorRelayAsync(string cameraId, string sourceUrl, CancellationToken ct)
    {
        try
        {
            var url = $"{SimulatorApiBaseUrl}/simulator/relays/{Uri.EscapeDataString(cameraId)}/start";
            var body = new SimulatorRelayStartBody
            {
                SourceUrl = sourceUrl,
                TranscodeH264 = true,
            };
            var res = await _http.PostAsJsonAsync(url, body, ct);
            if (res.IsSuccessStatusCode)
                return true;
            var txt = await res.Content.ReadAsStringAsync(ct);
            _logger.LogWarning(
                "Simulator relay start failed for {CameraId}: {Status} {Body}",
                cameraId,
                (int)res.StatusCode,
                txt.Length > 200 ? txt[..200] : txt);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Simulator relay unreachable for camera {CameraId}", cameraId);
            return false;
        }
    }

    private static bool IsMediaMtxSelfSource(string streamUrl)
    {
        if (!Uri.TryCreate(streamUrl.Trim(), UriKind.Absolute, out var u))
            return false;
        if (!string.Equals(u.Scheme, "rtsp", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(u.Scheme, "rtsps", StringComparison.OrdinalIgnoreCase))
            return false;
        var host = u.Host.Trim().ToLowerInvariant();
        if (host != "omni-mediamtx" && host != "localhost" && host != "127.0.0.1")
            return false;
        var port = u.IsDefaultPort ? 0 : u.Port;
        if (port != 0 && port != 8554 && port != 18554)
            return false;
        return u.AbsolutePath.Trim('/').Length > 0;
    }

    private sealed class MediaMtxPathPayload
    {
        [JsonPropertyName("source")]
        public string Source { get; set; } = "";

        [JsonPropertyName("sourceOnDemand")]
        public bool SourceOnDemand { get; set; }

        [JsonPropertyName("sourceOnDemandStartTimeout")]
        public string SourceOnDemandStartTimeout { get; set; } = "15s";

        [JsonPropertyName("sourceOnDemandCloseAfter")]
        public string SourceOnDemandCloseAfter { get; set; } = "5m";
    }

    private sealed class SimulatorRelayStartBody
    {
        [JsonPropertyName("source_url")]
        public string SourceUrl { get; set; } = "";

        [JsonPropertyName("transcode_h264")]
        public bool TranscodeH264 { get; set; } = true;
    }
}
