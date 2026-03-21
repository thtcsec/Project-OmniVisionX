using Microsoft.AspNetCore.Mvc;

namespace Omni.API.Controllers;

/// <summary>
/// Proxies omni-object <c>/rtsp/detections/latest</c> so the browser can poll latest bboxes
/// without CORS (same origin as API) when Redis→SignalR path is misconfigured.
/// </summary>
[ApiController]
[Route("api/live/detections")]
public class LiveDetectionsController : ControllerBase
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly IConfiguration _configuration;
    private readonly ILogger<LiveDetectionsController> _logger;

    public LiveDetectionsController(
        IHttpClientFactory httpFactory,
        IConfiguration configuration,
        ILogger<LiveDetectionsController> logger)
    {
        _httpFactory = httpFactory;
        _configuration = configuration;
        _logger = logger;
    }

    [HttpGet("latest")]
    public async Task<IActionResult> GetLatest([FromQuery] string? cameraIds, CancellationToken cancellationToken)
    {
        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? "http://localhost:8555").TrimEnd('/');
        var url = $"{baseUrl}/rtsp/detections/latest";
        if (!string.IsNullOrWhiteSpace(cameraIds))
            url += $"?cameraIds={Uri.EscapeDataString(cameraIds)}";

        try
        {
            var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(8);
            using var res = await client.GetAsync(url, cancellationToken);
            var body = await res.Content.ReadAsStringAsync(cancellationToken);
            if (!res.IsSuccessStatusCode)
            {
                _logger.LogWarning("Omni-object latest detections failed: {Status} {Body}", res.StatusCode, body);
                return StatusCode((int)res.StatusCode, body);
            }

            return Content(body, "application/json");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Omni-object unreachable at {Url}", url);
            return Ok(new { items = new Dictionary<string, object>(), error = "omni-object unreachable", detail = ex.Message });
        }
    }
}
