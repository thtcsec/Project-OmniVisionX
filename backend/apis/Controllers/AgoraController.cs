using AgoraIO.Media;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

/// <summary>
/// Agora RTC token — POST /api/agora/token
/// Uses official <see cref="RtcTokenBuilder2"/> (AccessToken2 / 007) from vendored AgoraIO/Tools.
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

    [HttpPost("token")]
    public IActionResult GenerateToken([FromBody] AgoraTokenRequest request)
    {
        var appId = NormalizeAgoraHexKey(_env.Get("OMNI_AGORA_APP_ID"));
        var appCert = NormalizeAgoraHexKey(_env.Get("OMNI_AGORA_APP_CERTIFICATE"));

        if (string.IsNullOrWhiteSpace(appId) || string.IsNullOrWhiteSpace(appCert))
            return StatusCode(412, new { error = "Agora not configured. Set OMNI_AGORA_APP_ID and OMNI_AGORA_APP_CERTIFICATE in Settings." });

        var channel = (request.ChannelName ?? "").Trim();
        if (channel.Length == 0)
            return BadRequest(new { error = "channelName is required." });

        var uid = request.Uid ?? 0;
        var expireSeconds = Math.Clamp(request.ExpireSeconds ?? 3600, 60, 86400);
        var isPublisher = string.Equals((request.Role ?? "").Trim(), "publisher", StringComparison.OrdinalIgnoreCase);

        var role = isPublisher ? RtcTokenBuilder2.Role.RolePublisher : RtcTokenBuilder2.Role.RoleSubscriber;
        var exp = (uint)expireSeconds;

        // tokenExpire + privilegeExpire: same window is fine for demo (see Agora sample code).
        var token = RtcTokenBuilder2.buildTokenWithUid(appId, appCert, channel, uid, role, exp, exp);
        if (string.IsNullOrEmpty(token))
            return StatusCode(500, new { error = "Failed to build Agora token. Check App ID (32-char hex) and Certificate match your Agora Console project." });

        return Ok(new { token, appId, channel, uid, expireSeconds });
    }

    [HttpGet("status")]
    public IActionResult Status()
    {
        var appId = NormalizeAgoraHexKey(_env.Get("OMNI_AGORA_APP_ID"));
        var certRaw = _env.Get("OMNI_AGORA_APP_CERTIFICATE");
        var hasCert = !string.IsNullOrWhiteSpace(certRaw);
        return Ok(new
        {
            configured = !string.IsNullOrWhiteSpace(appId) && hasCert,
            appId = string.IsNullOrWhiteSpace(appId) ? null : appId,
        });
    }

    /// <summary>Strip quotes and hyphens — Agora token builder expects 32-char hex (no UUID dashes).</summary>
    private static string NormalizeAgoraHexKey(string? s)
    {
        if (string.IsNullOrWhiteSpace(s)) return "";
        var t = s.Trim().Trim('"', '\'');
        if (t.Contains('-'))
            t = t.Replace("-", "", StringComparison.Ordinal);
        return t;
    }

    public sealed record AgoraTokenRequest(string? ChannelName, uint? Uid, string? Role, int? ExpireSeconds);
}
