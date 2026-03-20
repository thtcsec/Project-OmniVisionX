using Microsoft.AspNetCore.Mvc;
using Omni.API.Models;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DetectionsController : ControllerBase
{
    private readonly ILogger<DetectionsController> _logger;

    public DetectionsController(ILogger<DetectionsController> logger)
    {
        _logger = logger;
    }

    [HttpGet]
    public ActionResult<IEnumerable<Detection>> GetDetections(
        [FromQuery] string? cameraId = null,
        [FromQuery] DateTime? from = null,
        [FromQuery] DateTime? to = null,
        [FromQuery] int limit = 100)
    {
        // TODO: Query from PostgreSQL with filters
        return Ok(new List<Detection>());
    }

    [HttpGet("latest")]
    public ActionResult<IEnumerable<Detection>> GetLatestDetections([FromQuery] string? cameraIds = null)
    {
        // TODO: Get latest detections from Redis
        return Ok(new List<Detection>());
    }

    [HttpGet("stats")]
    public ActionResult GetStats([FromQuery] string? cameraId = null)
    {
        // TODO: Aggregate stats
        return Ok(new
        {
            totalDetections = 0,
            byClass = new { },
            byHour = new { }
        });
    }
}
