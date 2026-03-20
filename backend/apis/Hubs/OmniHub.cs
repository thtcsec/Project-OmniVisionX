using Microsoft.AspNetCore.SignalR;

namespace Omni.API.Hubs;

public class OmniHub : Hub
{
    private readonly ILogger<OmniHub> _logger;

    public OmniHub(ILogger<OmniHub> logger)
    {
        _logger = logger;
    }

    // Client joins camera group to receive events for specific camera
    public async Task JoinCameraGroup(string cameraId)
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, $"camera-{cameraId}");
        _logger.LogInformation("Client {ConnectionId} joined camera-{CameraId}", Context.ConnectionId, cameraId);
    }

    public async Task LeaveCameraGroup(string cameraId)
    {
        await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"camera-{cameraId}");
        _logger.LogInformation("Client {ConnectionId} left camera-{CameraId}", Context.ConnectionId, cameraId);
    }

    // Subscribe to all events
    public async Task SubscribeAll()
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, "omni-all");
        _logger.LogInformation("Client {ConnectionId} subscribed to all events", Context.ConnectionId);
    }

    public override async Task OnConnectedAsync()
    {
        _logger.LogInformation("Client connected: {ConnectionId}", Context.ConnectionId);
        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected: {ConnectionId}", Context.ConnectionId);
        await base.OnDisconnectedAsync(exception);
    }
}
