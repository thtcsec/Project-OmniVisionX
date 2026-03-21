using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using Omni.API.Data;
using Omni.API.Hubs;
using Omni.API.Models;
using Omni.API.Services;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CamerasController : ControllerBase
{
    private readonly OmniDbContext _db;
    private readonly IHubContext<OmniHub> _hub;
    private readonly MediaMtxPathRegistrar _mediaMtx;

    public CamerasController(OmniDbContext db, IHubContext<OmniHub> hub, MediaMtxPathRegistrar mediaMtx)
    {
        _db = db;
        _hub = hub;
        _mediaMtx = mediaMtx;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Camera>>> GetCameras()
    {
        return Ok(await _db.Cameras.AsNoTracking().ToListAsync());
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<Camera>> GetCamera(string id)
    {
        var camera = await _db.Cameras.AsNoTracking().FirstOrDefaultAsync(c => c.Id == id);
        if (camera == null) return NotFound();
        return Ok(camera);
    }

    [HttpPost]
    public async Task<ActionResult<Camera>> CreateCamera([FromBody] CameraWriteDto dto)
    {
        if (!ModelState.IsValid)
            return ValidationProblem(ModelState);

        var urlErr = ValidateStreamUrl(dto.StreamUrl);
        if (urlErr != null)
            return BadRequest(new { error = urlErr });

        var camera = new Camera
        {
            Id = Guid.NewGuid().ToString("N"),
            Name = dto.Name.Trim(),
            StreamUrl = dto.StreamUrl.Trim(),
            Status = dto.Status is "online" or "offline" ? dto.Status : "offline",
            EnableObjectDetection = dto.EnableObjectDetection,
            EnablePlateOcr = dto.EnablePlateOcr,
            EnableFaceRecognition = dto.EnableFaceRecognition,
        };

        _db.Cameras.Add(camera);
        await _db.SaveChangesAsync();

        await _hub.Clients.Group("omni-all").SendAsync("CamerasChanged", new { action = "created", id = camera.Id });

        if (string.Equals(camera.Status, "online", StringComparison.OrdinalIgnoreCase))
            await _mediaMtx.TryRegisterPathAsync(camera.Id, camera.StreamUrl);

        return CreatedAtAction(nameof(GetCamera), new { id = camera.Id }, camera);
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> UpdateCamera(string id, [FromBody] CameraWriteDto dto)
    {
        if (!ModelState.IsValid)
            return ValidationProblem(ModelState);

        var urlErr = ValidateStreamUrl(dto.StreamUrl);
        if (urlErr != null)
            return BadRequest(new { error = urlErr });

        var camera = await _db.Cameras.FindAsync(id);
        if (camera == null) return NotFound();

        camera.Name = dto.Name.Trim();
        camera.StreamUrl = dto.StreamUrl.Trim();
        camera.Status = dto.Status is "online" or "offline" ? dto.Status : camera.Status;
        camera.EnableObjectDetection = dto.EnableObjectDetection;
        camera.EnablePlateOcr = dto.EnablePlateOcr;
        camera.EnableFaceRecognition = dto.EnableFaceRecognition;

        await _db.SaveChangesAsync();

        await _hub.Clients.Group("omni-all").SendAsync("CamerasChanged", new { action = "updated", id });

        if (string.Equals(camera.Status, "online", StringComparison.OrdinalIgnoreCase))
            await _mediaMtx.TryRegisterPathAsync(camera.Id, camera.StreamUrl);

        return NoContent();
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> DeleteCamera(string id)
    {
        var camera = await _db.Cameras.FindAsync(id);
        if (camera == null) return NotFound();

        _db.Cameras.Remove(camera);
        await _db.SaveChangesAsync();

        await _hub.Clients.Group("omni-all").SendAsync("CamerasChanged", new { action = "deleted", id });

        return NoContent();
    }

    [HttpGet("{id}/stats")]
    public ActionResult<CameraStats> GetCameraStats(string id)
    {
        return Ok(new CameraStats
        {
            CameraId = id,
            Healthy = false,
            TotalDetections = 0,
            TotalPlates = 0,
            LastFrameTime = DateTime.UtcNow
        });
    }

    [HttpPost("{id}/subscribe")]
    public IActionResult SubscribeToCamera(string id)
    {
        return Ok(new { message = $"Use SignalR Hub to subscribe to camera {id}." });
    }

    private static string? ValidateStreamUrl(string url)
    {
        if (string.IsNullOrWhiteSpace(url))
            return null;
        var u = url.Trim();
        if (u.StartsWith("rtsp://", StringComparison.OrdinalIgnoreCase)
            || u.StartsWith("rtsps://", StringComparison.OrdinalIgnoreCase))
            return null;
        // Cho phép http(s) cho HLS preview sau này
        if (u.StartsWith("http://", StringComparison.OrdinalIgnoreCase)
            || u.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            return null;
        return "Stream URL must be rtsp(s):// or http(s)://";
    }
}
