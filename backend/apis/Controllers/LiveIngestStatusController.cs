using System.Text.Json;
using Microsoft.AspNetCore.Mvc;

namespace Omni.API.Controllers;

/// <summary>
/// Exposes omni-object capture-pool IDs so the UI can explain missing bbox / snapshot 404
/// (detector only ingests DB cameras with Status = online).
/// </summary>
[ApiController]
[Route("api/live/ingest-status")]
public sealed class LiveIngestStatusController : ControllerBase
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly IConfiguration _configuration;
    private readonly ILogger<LiveIngestStatusController> _logger;

    public LiveIngestStatusController(
        IHttpClientFactory httpFactory,
        IConfiguration configuration,
        ILogger<LiveIngestStatusController> logger)
    {
        _httpFactory = httpFactory;
        _configuration = configuration;
        _logger = logger;
    }

    [HttpGet]
    public async Task<IActionResult> Get(CancellationToken cancellationToken)
    {
        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? "http://localhost:8555").TrimEnd('/');

        try
        {
            var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(6);

            using var healthRes = await client.GetAsync($"{baseUrl}/rtsp/health", cancellationToken);
            var healthBody = await healthRes.Content.ReadAsStringAsync(cancellationToken);

            using var camsRes = await client.GetAsync($"{baseUrl}/rtsp/cameras", cancellationToken);
            var camsBody = await camsRes.Content.ReadAsStringAsync(cancellationToken);

            var ingestIds = new List<string>();
            if (camsRes.IsSuccessStatusCode)
            {
                try
                {
                    using var doc = JsonDocument.Parse(camsBody);
                    if (doc.RootElement.TryGetProperty("cameras", out var arr) &&
                        arr.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var el in arr.EnumerateArray())
                        {
                            if (el.TryGetProperty("camera_id", out var cid) &&
                                cid.ValueKind == JsonValueKind.String)
                            {
                                var s = cid.GetString();
                                if (!string.IsNullOrEmpty(s))
                                    ingestIds.Add(s);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to parse omni-object /rtsp/cameras JSON");
                }
            }

            return Ok(new
            {
                reachable = true,
                healthOk = healthRes.IsSuccessStatusCode,
                healthStatus = (int)healthRes.StatusCode,
                camerasStatus = (int)camsRes.StatusCode,
                ingestCameraIds = ingestIds,
            });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "omni-object unreachable at {Base}", baseUrl);
            return Ok(new
            {
                reachable = false,
                ingestCameraIds = Array.Empty<string>(),
                detail = ex.Message,
            });
        }
    }
}
