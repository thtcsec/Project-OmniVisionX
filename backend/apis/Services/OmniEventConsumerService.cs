using Microsoft.AspNetCore.SignalR;
using Omni.API.Hubs;
using StackExchange.Redis;

namespace Omni.API.Services;

public class OmniEventConsumerService : BackgroundService
{
    private readonly ILogger<OmniEventConsumerService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IHubContext<OmniHub> _hubContext;

    // Redis stream keys to consume
    private readonly string[] _streamKeys = { "omni:detections", "omni:vehicles", "omni:humans" };

    public OmniEventConsumerService(
        ILogger<OmniEventConsumerService> logger,
        IServiceProvider serviceProvider,
        IHubContext<OmniHub> hubContext)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _hubContext = hubContext;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("OmniEventConsumerService starting...");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var scope = _serviceProvider.CreateScope();
                var redis = scope.ServiceProvider.GetRequiredService<RedisService>();
                var db = redis.GetDatabase();

                // Scan for new messages using blocking read
                foreach (var streamKey in _streamKeys)
                {
                    try
                    {
                        var entries = await db.StreamReadAsync(streamKey, "omni-api-consumers", count: 10);

                        foreach (var entry in entries)
                        {
                            await ProcessEntry(streamKey, entry);
                            await db.StreamAcknowledgeAsync(streamKey, "omni-api-consumers", entry.Id);
                        }
                    }
                    catch (RedisServerException ex) when (ex.Message.Contains("NOGROUP"))
                    {
                        // Create consumer group if not exists
                        await db.StreamCreateConsumerGroupAsync(streamKey, "omni-api-consumers", "0", createStream: true);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning("Error reading stream {Stream}: {Error}", streamKey, ex.Message);
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

    private async Task ProcessEntry(string streamKey, StreamEntry entry)
    {
        var eventType = entry.Values.FirstOrDefault(e => e.Name == "event_type").Value.ToString();
        var cameraId = entry.Values.FirstOrDefault(e => e.Name == "camera_id").Value.ToString();
        var payload = entry.Values.FirstOrDefault(e => e.Name == "payload").Value.ToString();

        // Broadcast to camera group
        if (!string.IsNullOrEmpty(cameraId))
        {
            await _hubContext.Clients.Group($"camera-{cameraId}").SendAsync("OmniEvent", new
            {
                type = eventType,
                cameraId,
                data = payload,
                timestamp = DateTime.UtcNow
            });
        }

        // Broadcast to all subscribers
        await _hubContext.Clients.Group("omni-all").SendAsync("OmniEvent", new
        {
            type = eventType,
            cameraId,
            data = payload,
            timestamp = DateTime.UtcNow
        });
    }
}
