using Microsoft.EntityFrameworkCore;
using Omni.API.Persistence;

namespace Omni.API.Services;

/// <summary>On startup, register all online RTSP cameras with MediaMTX (paths match UI).</summary>
public sealed class MediaMtxStartupSync : IHostedService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<MediaMtxStartupSync> _logger;

    public MediaMtxStartupSync(IServiceScopeFactory scopeFactory, ILogger<MediaMtxStartupSync> logger)
    {
        _scopeFactory = scopeFactory;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await Task.Yield();
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var registrar = scope.ServiceProvider.GetService<MediaMtxPathRegistrar>();
            if (registrar is not { IsConfigured: true })
                return;

            var db = scope.ServiceProvider.GetRequiredService<OmniDbContext>();
            var cameras = await db.Cameras.AsNoTracking()
                .Where(c => c.Status == "online" && c.StreamUrl != null && c.StreamUrl != "")
                .ToListAsync(cancellationToken);

            foreach (var c in cameras)
            {
                if (!c.StreamUrl.Trim().StartsWith("rtsp", StringComparison.OrdinalIgnoreCase))
                    continue;
                await registrar.TryRegisterPathAsync(c.Id, c.StreamUrl, cancellationToken);
            }

            _logger.LogInformation("MediaMTX startup sync: processed {Count} online camera(s)", cameras.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "MediaMTX startup sync failed (UI preview may need a camera save/restart)");
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
