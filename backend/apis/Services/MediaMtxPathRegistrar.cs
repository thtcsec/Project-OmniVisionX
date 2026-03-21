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

    /// <returns>true if skipped (not RTSP / offline / no config) or HTTP 200/204.</returns>
    public async Task<bool> TryRegisterPathAsync(string cameraId, string streamUrl, CancellationToken ct = default)
    {
        var baseUrl = ApiBaseUrl;
        if (string.IsNullOrEmpty(baseUrl))
            return true;

        if (string.IsNullOrWhiteSpace(streamUrl)
            || !streamUrl.Trim().StartsWith("rtsp", StringComparison.OrdinalIgnoreCase))
            return true;

        var path = Uri.EscapeDataString(cameraId);
        var url = $"{baseUrl}/v3/config/paths/replace/{path}";
        var payload = new MediaMtxPathPayload
        {
            Source = streamUrl.Trim(),
            SourceOnDemand = true,
            SourceOnDemandStartTimeout = "15s",
            SourceOnDemandCloseAfter = "5m",
        };

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
}
