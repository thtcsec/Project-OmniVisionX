using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Npgsql;
using Omni.API.Data;
using Omni.API.Models;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PlatesController : ControllerBase
{
    private readonly ILogger<PlatesController> _logger;
    private readonly OmniDbContext _db;

    public PlatesController(ILogger<PlatesController> logger, OmniDbContext db)
    {
        _logger = logger;
        _db = db;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<PlateHitDto>>> GetPlates(
        [FromQuery] string? cameraId = null,
        [FromQuery] string? plateText = null,
        [FromQuery] DateTime? from = null,
        [FromQuery] DateTime? to = null,
        [FromQuery] int limit = 100,
        CancellationToken ct = default)
    {
        limit = Math.Clamp(limit, 1, 500);
        try
        {
            var q = _db.PlateRecords
                .AsNoTracking()
                .Where(x => !x.IsDeleted);

            if (!string.IsNullOrWhiteSpace(cameraId))
                q = q.Where(x => x.CameraId == cameraId);

            if (!string.IsNullOrWhiteSpace(plateText))
            {
                var needle = plateText.Trim().ToUpperInvariant();
                q = q.Where(x => x.PlateNumber.ToUpper().Contains(needle));
            }

            if (from is not null)
                q = q.Where(x => x.Timestamp >= from.Value);
            if (to is not null)
                q = q.Where(x => x.Timestamp <= to.Value);

            var cameras = _db.Cameras.AsNoTracking();

            var result = await (
                from p in q
                join c in cameras on p.CameraId equals c.Id into cam
                from c in cam.DefaultIfEmpty()
                orderby p.Timestamp descending
                select new PlateHitDto(
                    p.Id.ToString(),
                    p.CameraId,
                    c != null ? c.Name : null,
                    p.PlateNumber,
                    p.Confidence,
                    p.Timestamp,
                    p.VehicleType,
                    p.Color,
                    p.ThumbnailPath,
                    p.FullFramePath
                )
            ).Take(limit).ToListAsync(ct);

            return Ok(result);
        }
        catch (PostgresException ex) when (ex.SqlState == "42P01")
        {
            _logger.LogWarning("PlateRecords table missing. Returning empty list.");
            return Ok(Array.Empty<PlateHitDto>());
        }
    }

    [HttpGet("search")]
    public async Task<ActionResult<IEnumerable<PlateHitDto>>> SearchPlates(
        [FromQuery] string query,
        [FromQuery] int limit = 50,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(query))
            return Ok(Array.Empty<PlateHitDto>());
        return await GetPlates(plateText: query, limit: limit, ct: ct);
    }

    [HttpGet("stats")]
    public async Task<ActionResult> GetStats([FromQuery] string? cameraId = null, CancellationToken ct = default)
    {
        var q = _db.PlateRecords.AsNoTracking().Where(x => !x.IsDeleted);
        if (!string.IsNullOrWhiteSpace(cameraId))
            q = q.Where(x => x.CameraId == cameraId);

        var total = await q.CountAsync(ct);
        var unique = await q.Select(x => x.PlateNumber).Distinct().CountAsync(ct);
        var top = await q.GroupBy(x => x.PlateNumber)
            .Select(g => new { plate = g.Key, count = g.Count() })
            .OrderByDescending(x => x.count)
            .Take(10)
            .ToListAsync(ct);
        var byType = await q.GroupBy(x => x.VehicleType)
            .Select(g => new { vehicleType = g.Key, count = g.Count() })
            .OrderByDescending(x => x.count)
            .ToListAsync(ct);

        return Ok(new
        {
            totalPlates = total,
            uniquePlates = unique,
            topPlates = top,
            byVehicleType = byType,
        });
    }

    public sealed record PlateHitDto(
        string Id,
        string CameraId,
        string? CameraName,
        string PlateText,
        float Confidence,
        DateTime Timestamp,
        string VehicleType,
        string? Color,
        string? PlateImageUrl,
        string? FrameImageUrl);
}
