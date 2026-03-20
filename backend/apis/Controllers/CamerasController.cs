using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using Omni.API.Data;
using Omni.API.Hubs;
using Omni.API.Models;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CamerasController : ControllerBase
{
    private readonly OmniDbContext _db;
    private readonly IHubContext<OmniHub> _hub;

    public CamerasController(OmniDbContext db, IHubContext<OmniHub> hub)
    {
        _db = db;
        _hub = hub;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Camera>>> GetCameras()
    {
        var cameras = _db.Cameras.ToList();
        return Ok(cameras);
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<Camera>> GetCamera(string id)
    {
        var camera = await _db.Cameras.FindAsync(id);
        if (camera == null) return NotFound();
        return Ok(camera);
    }

    [HttpPost]
    public async Task<ActionResult<Camera>> CreateCamera([FromBody] Camera camera)
    {
        _db.Cameras.Add(camera);
        await _db.SaveChangesAsync();
        return CreatedAtAction(nameof(GetCamera), new { id = camera.Id }, camera);
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> UpdateCamera(string id, [FromBody] Camera camera)
    {
        if (id != camera.Id) return BadRequest();
        _db.Entry(camera).State = EntityState.Modified;
        await _db.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> DeleteCamera(string id)
    {
        var camera = await _db.Cameras.FindAsync(id);
        if (camera == null) return NotFound();
        _db.Cameras.Remove(camera);
        await _db.SaveChangesAsync();
        return NoContent();
    }

    [HttpGet("{id}/stats")]
    public ActionResult<CameraStats> GetCameraStats(string id)
    {
        // TODO: Get stats from Redis
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
    public async Task<IActionResult> SubscribeToCamera(string id)
    {
        // SignalR group management is handled in Hub
        await _hub.Groups.AddToGroupAsync(Context.ConnectionId, $"camera-{id}");
        return Ok(new { message = $"Subscribed to camera {id}" });
    }
}
