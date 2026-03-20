using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Omni.API.Data;
using StackExchange.Redis;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class StatsController : ControllerBase
{
    private readonly OmniDbContext _db;
    private readonly Services.RedisService _redis;

    public StatsController(OmniDbContext db, Services.RedisService redisService)
    {
        _db = db;
        _redis = redisService;
    }

    [HttpGet("dashboard")]
    public async Task<IActionResult> GetDashboardStats()
    {
        int totalCameras = 0;
        int onlineCameras = 0;

        try 
        {
            totalCameras = await _db.Cameras.CountAsync();
            onlineCameras = await _db.Cameras.CountAsync(c => c.Status == "online");
        }
        catch 
        {
            // Fallback if table doesn't exist
        }

        long detectionsToday = 0;
        long platesDetected = 0;
        long activeAlerts = 0;

        try
        {
            detectionsToday = await _redis.StreamLengthAsync("omni:detections");
            platesDetected = await _redis.StreamLengthAsync("omni:vehicles");
        }
        catch
        {
            // Ignore Redis errors
        }

        return Ok(new
        {
            camerasOnline = onlineCameras,
            camerasTotal = totalCameras,
            detectionsToday = detectionsToday,
            platesDetected = platesDetected,
            activeAlerts = activeAlerts
        });
    }
}
