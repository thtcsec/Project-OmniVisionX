using System.Text.Json;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Configuration;
using Omni.API.Hubs;
using StackExchange.Redis;

namespace Omni.API.Services;

/// <summary>
/// Consumes Redis Streams (XREADGROUP) and pushes SignalR <c>OmniEvent</c> to clients.
/// Python publishers use fields: camera_id, class_name, bbox, confidence, timestamp, etc.
/// </summary>
public class OmniEventConsumerService : BackgroundService
{
    private const string ConsumerGroup = "omni-api-consumers";
    private readonly string _consumerName =
        $"api-{Environment.MachineName}-{Environment.ProcessId}";

    private readonly ILogger<OmniEventConsumerService> _logger;
    private readonly RedisService _redis;
    private readonly IHubContext<OmniHub> _hubContext;

    private readonly string[] _streamKeys;

    public OmniEventConsumerService(
        ILogger<OmniEventConsumerService> logger,
        RedisService redis,
        IHubContext<OmniHub> hubContext,
        IConfiguration configuration)
    {
        _logger = logger;
        _redis = redis;
        _hubContext = hubContext;
        var prefix = (configuration["Omni:RedisStreamPrefix"] ?? "omni").Trim();
        if (string.IsNullOrEmpty(prefix))
            prefix = "omni";
        _streamKeys = new[]
        {
            $"{prefix}:detections",
            $"{prefix}:vehicles",
            $"{prefix}:humans",
        };
        _logger.LogInformation("OmniEventConsumerService stream prefix: {Prefix}", prefix);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("OmniEventConsumerService starting (consumer {Consumer})…", _consumerName);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var db = _redis.GetDatabase();

                foreach (var streamKey in _streamKeys)
                {
                    try
                    {
                        await EnsureConsumerGroupAsync(db, streamKey);

                        // ">" = only new messages for this consumer group (XREADGROUP … STREAMS key >)
                        var entries = await db.StreamReadGroupAsync(
                            streamKey,
                            ConsumerGroup,
                            _consumerName,
                            ">",
                            count: 25);

                        foreach (var entry in entries)
                        {
                            await ProcessEntryAsync(streamKey, entry);
                            await db.StreamAcknowledgeAsync(streamKey, ConsumerGroup, entry.Id);
                        }
                    }
                    catch (RedisServerException ex) when (ex.Message.Contains("NOGROUP", StringComparison.OrdinalIgnoreCase))
                    {
                        await EnsureConsumerGroupAsync(db, streamKey);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Error reading stream {Stream}: {Message}", streamKey, ex.Message);
                    }
                }

                await Task.Delay(100, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in OmniEventConsumerService");
                await Task.Delay(1000, stoppingToken);
            }
        }
    }

    private async Task EnsureConsumerGroupAsync(IDatabase db, string streamKey)
    {
        try
        {
            // "$" = group starts at tail — avoid flooding UI with full history on first deploy
            await db.StreamCreateConsumerGroupAsync(streamKey, ConsumerGroup, "$", createStream: true);
            _logger.LogInformation("Created consumer group {Group} on {Stream}", ConsumerGroup, streamKey);
        }
        catch (RedisServerException ex) when (ex.Message.Contains("BUSYGROUP", StringComparison.OrdinalIgnoreCase))
        {
            // Already exists
        }
    }

    private async Task ProcessEntryAsync(string streamKey, StreamEntry entry)
    {
        var fields = entry.Values.ToDictionary(
            v => v.Name.ToString(),
            v => v.Value.ToString() ?? string.Empty,
            StringComparer.Ordinal);

        var cameraId = fields.GetValueOrDefault("camera_id") ?? "";

        var eventType = ResolveEventType(streamKey, fields);

        var payload = BuildPayloadJson(streamKey, fields);

        var omniEvent = new
        {
            type = eventType,
            cameraId,
            data = payload,
            timestamp = DateTime.UtcNow.ToString("o"),
        };

        if (!string.IsNullOrEmpty(cameraId))
        {
            await _hubContext.Clients.Group($"camera-{cameraId}").SendAsync("OmniEvent", omniEvent);
        }

        await _hubContext.Clients.Group("omni-all").SendAsync("OmniEvent", omniEvent);
    }

    private static string ResolveEventType(string streamKey, Dictionary<string, string> fields)
    {
        if (streamKey.EndsWith(":vehicles", StringComparison.OrdinalIgnoreCase))
        {
            if (!string.IsNullOrEmpty(fields.GetValueOrDefault("plate_text")))
                return "plate";
            return "vehicle";
        }

        if (streamKey.EndsWith(":humans", StringComparison.OrdinalIgnoreCase))
            return "human";

        return "detection";
    }

    private static string BuildPayloadJson(string streamKey, Dictionary<string, string> fields)
    {
        // Match EventFeed: label, plateText, type
        object payload = streamKey switch
        {
            var s when s.EndsWith(":vehicles", StringComparison.OrdinalIgnoreCase) => new
            {
                label = fields.GetValueOrDefault("class_name"),
                plateText = fields.GetValueOrDefault("plate_text"),
                bbox = fields.GetValueOrDefault("bbox"),
                confidence = fields.GetValueOrDefault("confidence"),
                plateConfidence = fields.GetValueOrDefault("plate_confidence"),
                vehicleColor = fields.GetValueOrDefault("vehicle_color"),
                timestamp = fields.GetValueOrDefault("timestamp"),
            },
            var s when s.EndsWith(":humans", StringComparison.OrdinalIgnoreCase) => new
            {
                label = fields.GetValueOrDefault("class_name"),
                faceIdentity = fields.GetValueOrDefault("face_identity"),
                bbox = fields.GetValueOrDefault("bbox"),
                confidence = fields.GetValueOrDefault("confidence"),
                faceConfidence = fields.GetValueOrDefault("face_confidence"),
                timestamp = fields.GetValueOrDefault("timestamp"),
            },
            _ => new
            {
                label = fields.GetValueOrDefault("class_name"),
                globalTrackId = fields.GetValueOrDefault("global_track_id"),
                bbox = fields.GetValueOrDefault("bbox"),
                confidence = fields.GetValueOrDefault("confidence"),
                timestamp = fields.GetValueOrDefault("timestamp"),
                frameWidth = fields.GetValueOrDefault("frame_width"),
                frameHeight = fields.GetValueOrDefault("frame_height"),
            },
        };

        return JsonSerializer.Serialize(payload);
    }
}
