using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

/// <summary>
/// Agora RTC token generator — POST /api/agora/token
/// Implements Agora AccessToken2 (RTC) spec without third-party NuGet packages.
/// </summary>
[ApiController]
[Route("api/agora")]
public sealed class AgoraController : ControllerBase
{
    private readonly EnvFileReader _env;

    public AgoraController(EnvFileReader env)
    {
        _env = env;
    }

    // POST /api/agora/token
    [HttpPost("token")]
    public IActionResult GenerateToken([FromBody] AgoraTokenRequest request)
    {
        var appId = _env.Get("OMNI_AGORA_APP_ID");
        var appCert = _env.Get("OMNI_AGORA_APP_CERTIFICATE");

        if (string.IsNullOrWhiteSpace(appId) || string.IsNullOrWhiteSpace(appCert))
            return StatusCode(412, new { error = "Agora not configured. Set OMNI_AGORA_APP_ID and OMNI_AGORA_APP_CERTIFICATE in Settings." });

        var channel = (request.ChannelName ?? "").Trim();
        if (channel.Length == 0)
            return BadRequest(new { error = "channelName is required." });

        var uid = request.Uid ?? 0;
        var expireSeconds = Math.Clamp(request.ExpireSeconds ?? 3600, 60, 86400);
        var role = (request.Role ?? "subscriber").Trim().ToLowerInvariant() == "publisher"
            ? AgoraRole.Publisher
            : AgoraRole.Subscriber;

        var token = BuildRtcToken(appId, appCert, channel, uid, role, expireSeconds);
        return Ok(new { token, appId, channel, uid, expireSeconds });
    }

    // GET /api/agora/status
    [HttpGet("status")]
    public IActionResult Status()
    {
        var appId = _env.Get("OMNI_AGORA_APP_ID");
        var hasCert = _env.Has("OMNI_AGORA_APP_CERTIFICATE");
        return Ok(new
        {
            configured = !string.IsNullOrWhiteSpace(appId) && hasCert,
            appId = string.IsNullOrWhiteSpace(appId) ? null : appId,
        });
    }

    // ── Agora AccessToken2 (RTC) — pure C# implementation ───────────────
    private static string BuildRtcToken(
        string appId, string appCertificate,
        string channelName, uint uid,
        AgoraRole role, int expireSeconds)
    {
        var nowTs = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var expireTs = nowTs + (uint)expireSeconds;
        var salt = (uint)Random.Shared.Next(1, int.MaxValue);

        var privileges = new Dictionary<ushort, uint> { [1] = expireTs };
        if (role == AgoraRole.Publisher)
        {
            privileges[2] = expireTs;
            privileges[3] = expireTs;
        }

        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write(salt);
        bw.Write(expireTs);
        bw.Write((ushort)privileges.Count);
        foreach (var kv in privileges.OrderBy(x => x.Key)) { bw.Write(kv.Key); bw.Write(kv.Value); }
        bw.Flush();
        var msg = ms.ToArray();

        var uidStr = uid == 0 ? "" : uid.ToString();
        var toSign = Encoding.UTF8.GetBytes(appId + channelName + uidStr).Concat(msg).ToArray();
        var sig = HMACSHA256.HashData(Encoding.UTF8.GetBytes(appCertificate), toSign);

        var tokenBytes = PackToken(sig, msg);
        return "007" + Convert.ToBase64String(tokenBytes);
    }

    private static byte[] PackToken(byte[] sig, byte[] msg)
    {
        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write((ushort)sig.Length);
        bw.Write(sig);
        bw.Write(msg);
        bw.Flush();
        return ms.ToArray();
    }

    private enum AgoraRole { Publisher, Subscriber }

    public sealed record AgoraTokenRequest(string? ChannelName, uint? Uid, string? Role, int? ExpireSeconds);
}
