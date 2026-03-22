using System.Globalization;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Models;
using Omni.API.Services;
using StackExchange.Redis;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DetectionsController : ControllerBase
{
    private readonly ILogger<DetectionsController> _logger;
    private readonly RedisService _redis;
    private readonly IConfiguration _configuration;

    public DetectionsController(
        ILogger<DetectionsController> logger,
        RedisService redis,
        IConfiguration configuration)
    {
        _logger = logger;
        _redis = redis;
        _configuration = configuration;
    }

    /// <summary>Recent detection events from Redis Streams (omni:detections, omni:vehicles). Not persisted in PostgreSQL.</summary>
    [HttpGet]
    public async Task<ActionResult<IEnumerable<DetectionHistoryItem>>> GetDetections(
        [FromQuery] string? cameraId = null,
        [FromQuery] DateTime? from = null,
        [FromQuery] DateTime? to = null,
        [FromQuery] int limit = 100,
        CancellationToken ct = default)
    {
        limit = Math.Clamp(limit, 1, 500);
        var items = await BuildHistoryAsync(perStreamCap: Math.Max(limit * 3, 200), ct);
        IEnumerable<DetectionHistoryItem> q = items;
        if (!string.IsNullOrWhiteSpace(cameraId))
            q = q.Where(x => string.Equals(x.CameraId, cameraId.Trim(), StringComparison.Ordinal));
        if (from is not null)
            q = q.Where(x => ParseIso(x.Timestamp) >= from.Value);
        if (to is not null)
            q = q.Where(x => ParseIso(x.Timestamp) <= to.Value);
        var list = q.Take(limit).ToList();
        return Ok(list);
    }

    [HttpGet("latest")]
    public async Task<ActionResult<IEnumerable<DetectionHistoryItem>>> GetLatestDetections(
        [FromQuery] string? cameraIds = null,
        [FromQuery] int limit = 50,
        CancellationToken ct = default)
    {
        limit = Math.Clamp(limit, 1, 200);
        HashSet<string>? filter = null;
        if (!string.IsNullOrWhiteSpace(cameraIds))
        {
            filter = new HashSet<string>(
                cameraIds.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                StringComparer.OrdinalIgnoreCase);
        }

        var items = await BuildHistoryAsync(perStreamCap: Math.Max(limit * 4, 150), ct);
        var q = items.AsEnumerable();
        if (filter is not null)
            q = q.Where(x => filter.Contains(x.CameraId));
        return Ok(q.Take(limit).ToList());
    }

    [HttpGet("stats")]
    public async Task<ActionResult> GetStats([FromQuery] string? cameraId = null, CancellationToken ct = default)
    {
        var all = await BuildHistoryAsync(perStreamCap: 2000, ct);
        var list = (string.IsNullOrWhiteSpace(cameraId)
                ? all
                : all.Where(x => string.Equals(x.CameraId, cameraId.Trim(), StringComparison.Ordinal)))
            .ToList();
        var byClass = list
            .GroupBy(x => x.Type)
            .Select(g => new { type = g.Key, count = g.Count() })
            .OrderByDescending(x => x.count)
            .ToList();
        return Ok(new
        {
            totalDetections = list.Count,
            byClass,
        });
    }

    private async Task<List<DetectionHistoryItem>> BuildHistoryAsync(int perStreamCap, CancellationToken ct)
    {
        var prefix = (_configuration["Omni:RedisStreamPrefix"] ?? "omni").Trim();
        if (string.IsNullOrEmpty(prefix))
            prefix = "omni";

        var detectionsKey = $"{prefix}:detections";
        var vehiclesKey = $"{prefix}:vehicles";

        var detEntries = await _redis.StreamRevRangeAsync(detectionsKey, perStreamCap);
        var vehEntries = await _redis.StreamRevRangeAsync(vehiclesKey, perStreamCap);

        var list = new List<DetectionHistoryItem>(detEntries.Length + vehEntries.Length);
        foreach (var e in detEntries)
        {
            var row = ParseDetectionEntry(e, "detections");
            if (row is not null)
                list.Add(row);
        }

        foreach (var e in vehEntries)
        {
            var row = ParseVehicleEntry(e, "vehicles");
            if (row is not null)
                list.Add(row);
        }

        return list
            .OrderByDescending(x => ParseIso(x.Timestamp))
            .ToList();
    }

    private static DetectionHistoryItem? ParseDetectionEntry(StreamEntry e, string source)
    {
        var v = e.Values;
        var cam = GetField(v, "camera_id");
        if (string.IsNullOrWhiteSpace(cam))
            return null;
        var cls = GetField(v, "class_name") ?? "unknown";
        var conf = ParseFloat(GetField(v, "confidence"), 0f);
        var tid = ParseInt(GetField(v, "global_track_id"));
        var ts = ParseTimestamp(GetField(v, "timestamp"));
        var bbox = ParseBboxCsv(GetField(v, "bbox"));

        return new DetectionHistoryItem
        {
            Id = $"{source}:{e.Id}",
            CameraId = cam,
            Type = cls,
            Confidence = conf,
            Timestamp = ts.ToString("o"),
            TrackId = tid,
            Bbox = bbox,
            Source = source,
        };
    }

    private static DetectionHistoryItem? ParseVehicleEntry(StreamEntry e, string source)
    {
        var v = e.Values;
        var cam = GetField(v, "camera_id");
        if (string.IsNullOrWhiteSpace(cam))
            return null;
        var cls = GetField(v, "class_name") ?? "vehicle";
        var plate = GetField(v, "plate_text")?.Trim();
        var conf = ParseFloat(GetField(v, "confidence"), ParseFloat(GetField(v, "plate_confidence"), 0f));
        var tid = ParseInt(GetField(v, "global_track_id"));
        var ts = ParseTimestamp(GetField(v, "timestamp"));
        var bbox = ParseBboxCsv(GetField(v, "bbox"));

        var type = string.IsNullOrEmpty(plate) ? cls : $"{cls} · {plate}";
        var meta = new Dictionary<string, string>();
        if (!string.IsNullOrEmpty(plate))
            meta["plateText"] = plate;

        return new DetectionHistoryItem
        {
            Id = $"{source}:{e.Id}",
            CameraId = cam,
            Type = type,
            Confidence = conf,
            Timestamp = ts.ToString("o"),
            TrackId = tid,
            Bbox = bbox,
            Source = source,
            Metadata = meta.Count > 0 ? meta : null,
        };
    }

    private static string? GetField(NameValueEntry[] vals, string name)
    {
        foreach (var x in vals)
        {
            if (x.Name.IsNullOrEmpty) continue;
            if (string.Equals(x.Name.ToString(), name, StringComparison.OrdinalIgnoreCase))
                return x.Value.ToString();
        }

        return null;
    }

    private static float ParseFloat(string? s, float fallback)
    {
        if (string.IsNullOrWhiteSpace(s)) return fallback;
        return float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;
    }

    private static int? ParseInt(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return null;
        return int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : null;
    }

    private static DateTimeOffset ParseTimestamp(string? s)
    {
        if (string.IsNullOrWhiteSpace(s))
            return DateTimeOffset.UtcNow;
        if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var unix))
            return DateTimeOffset.FromUnixTimeMilliseconds((long)(unix * 1000.0));
        return DateTimeOffset.UtcNow;
    }

    private static DateTime ParseIso(string iso)
    {
        return DateTime.TryParse(iso, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out var dt)
            ? dt
            : DateTime.MinValue;
    }

    private static BboxNormDto? ParseBboxCsv(string? csv)
    {
        if (string.IsNullOrWhiteSpace(csv)) return null;
        var parts = csv.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 4) return null;
        if (!int.TryParse(parts[0], out var x1)) return null;
        if (!int.TryParse(parts[1], out var y1)) return null;
        if (!int.TryParse(parts[2], out var x2)) return null;
        if (!int.TryParse(parts[3], out var y2)) return null;
        return new BboxNormDto { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2 };
    }
}
