using Microsoft.AspNetCore.Mvc;
using Omni.API.Models;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PlatesController : ControllerBase
{
    private readonly ILogger<PlatesController> _logger;

    public PlatesController(ILogger<PlatesController> logger)
    {
        _logger = logger;
    }

    [HttpGet]
    public ActionResult<IEnumerable<PlateEvent>> GetPlates(
        [FromQuery] string? cameraId = null,
        [FromQuery] string? plateText = null,
        [FromQuery] DateTime? from = null,
        [FromQuery] DateTime? to = null,
        [FromQuery] int limit = 100)
    {
        // TODO: Query from PostgreSQL
        return Ok(new List<PlateEvent>());
    }

    [HttpGet("search")]
    public ActionResult<IEnumerable<PlateEvent>> SearchPlates(
        [FromQuery] string query,
        [FromQuery] int limit = 50)
    {
        // TODO: Full-text search on plate text
        return Ok(new List<PlateEvent>());
    }

    [HttpGet("stats")]
    public ActionResult GetStats([FromQuery] string? cameraId = null)
    {
        // TODO: Aggregate plate stats
        return Ok(new
        {
            totalPlates = 0,
            uniquePlates = 0,
            topPlates = new { },
            byVehicleType = new { }
        });
    }
}
