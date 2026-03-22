using Microsoft.AspNetCore.Mvc;

namespace Omni.API.Controllers;

/// <summary>
/// Proxies omni-object <c>/rtsp/snap/{cameraId}</c> so the browser can show the same frame
/// used for vision chat without CORS issues (same origin as API).
/// </summary>
[ApiController]
[Route("api/live/snapshot")]
public sealed class LiveSnapshotController : ControllerBase
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly IConfiguration _configuration;
    private readonly ILogger<LiveSnapshotController> _logger;

    public LiveSnapshotController(
        IHttpClientFactory httpFactory,
        IConfiguration configuration,
        ILogger<LiveSnapshotController> logger)
    {
        _httpFactory = httpFactory;
        _configuration = configuration;
        _logger = logger;
    }

    [HttpGet("{cameraId}")]
    [ResponseCache(NoStore = true, Location = ResponseCacheLocation.None)]
    public async Task<IActionResult> GetSnapshot(string cameraId, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(cameraId))
            return BadRequest();

        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? "http://localhost:8555").TrimEnd('/');
        var url = $"{baseUrl}/rtsp/snap/{Uri.EscapeDataString(cameraId.Trim())}";

        try
        {
            var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(12);
            using var res = await client.GetAsync(url, cancellationToken);
            var bytes = await res.Content.ReadAsByteArrayAsync(cancellationToken);
            if (!res.IsSuccessStatusCode)
            {
                _logger.LogWarning("Omni-object snap failed: {Status} {Len} bytes", res.StatusCode, bytes.Length);
                return StatusCode((int)res.StatusCode);
            }

            return File(bytes, "image/jpeg");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Omni-object snap unreachable at {Url}", url);
            return StatusCode(502, new { error = "omni-object unreachable", detail = ex.Message });
        }
    }
}
