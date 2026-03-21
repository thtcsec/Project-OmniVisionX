using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

/// <summary>
/// Agora RTC token generator — POST /api/agora/token
/// Implements Agora AccessToken2 (RTC) spec:
///   token = "007" + base64( pack_uint16(sig_len) + sig + zlib_compress(msg) )
///   msg   = pack(salt, expireTs, privilege_count, [privilege_key, expireTs]...)
///   sig   = HMAC-SHA256( appCertificate, appId + channelName + uidStr + msg )
/// All integers are little-endian (BinaryWriter default on .NET).
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
        var appId = _env.Get("OMNI_AGORA_APP_ID");
        var appCert = _env.Get("OMNI_AGORA_APP_CERTIFICATE");

        if (string.IsNullOrWhiteSpace(appId) || string.IsNullOrWhiteSpace(appCert))
            return StatusCode(412, new { error = "Agora not configured. Set OMNI_AGORA_APP_ID and OMNI_AGORA_APP_CERTIFICATE in Settings." });

        var channel = (request.ChannelName ?? "").Trim();
        if (channel.Length == 0)
            return BadRequest(new { error = "channelName is required." });

        var uid = request.Uid ?? 0;
        var expireSeconds = Math.Clamp(request.ExpireSeconds ?? 3600, 60, 86400);
        var isPublisher = string.Equals((request.Role ?? "").Trim(), "publisher", StringComparison.OrdinalIgnoreCase);

        var token = BuildRtcToken(appId, appCert, channel, uid, isPublisher, expireSeconds);
        return Ok(new { token, appId, channel, uid, expireSeconds });
    }

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

    // ── Agora AccessToken2 (RTC) ─────────────────────────────────────────
    // Reference: https://github.com/AgoraIO/Tools/tree/master/DynamicKey/AgoraDynamicKey
    // Privilege keys (RTC): 1=JOIN_CHANNEL, 2=PUB_AUDIO, 3=PUB_VIDEO, 7=SUB_ALL
    private static string BuildRtcToken(
        string appId, string appCertificate,
        string channelName, uint uid,
        bool isPublisher, int expireSeconds)
    {
        var nowTs = (uint)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var expireTs = nowTs + (uint)expireSeconds;
        var salt = (uint)Random.Shared.Next(1, int.MaxValue);

        // Build privilege map
        var privileges = new SortedDictionary<ushort, uint> { [1] = expireTs };
        if (isPublisher)
        {
            privileges[2] = expireTs; // pub audio
            privileges[3] = expireTs; // pub video
        }
        else
        {
            privileges[7] = expireTs; // sub all streams
        }

        // Pack message: salt(4) + expireTs(4) + count(2) + [key(2)+val(4)]...
        var msg = PackMessage(salt, expireTs, privileges);

        // Signature: HMAC-SHA256(cert, appId + channel + uidStr + msg)
        var uidStr = uid == 0 ? "" : uid.ToString();
        var sigInput = Concat(
            Encoding.UTF8.GetBytes(appId),
            Encoding.UTF8.GetBytes(channelName),
            Encoding.UTF8.GetBytes(uidStr),
            msg);
        var sig = HMACSHA256.HashData(Encoding.UTF8.GetBytes(appCertificate), sigInput);

        // Compress msg with zlib (deflate with 2-byte zlib header)
        var compressedMsg = ZlibCompress(msg);

        // Pack token: sig_len(2LE) + sig(32) + compressed_msg
        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write((ushort)sig.Length);
        bw.Write(sig);
        bw.Write(compressedMsg);
        bw.Flush();

        return "007" + Convert.ToBase64String(ms.ToArray());
    }

    private static byte[] PackMessage(uint salt, uint expireTs, SortedDictionary<ushort, uint> privileges)
    {
        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        bw.Write(salt);
        bw.Write(expireTs);
        bw.Write((ushort)privileges.Count);
        foreach (var kv in privileges)
        {
            bw.Write(kv.Key);
            bw.Write(kv.Value);
        }
        bw.Flush();
        return ms.ToArray();
    }

    private static byte[] ZlibCompress(byte[] data)
    {
        // zlib = 2-byte header (0x78 0x9C) + deflate + 4-byte Adler-32
        using var output = new System.IO.MemoryStream();
        // Write zlib header (CM=8, CINFO=7 → 0x78; FCHECK for no dict, default compression → 0x9C)
        output.WriteByte(0x78);
        output.WriteByte(0x9C);
        using (var deflate = new DeflateStream(output, CompressionLevel.Optimal, leaveOpen: true))
        {
            deflate.Write(data, 0, data.Length);
        }
        // Adler-32 checksum (big-endian)
        var adler = Adler32(data);
        output.WriteByte((byte)(adler >> 24));
        output.WriteByte((byte)(adler >> 16));
        output.WriteByte((byte)(adler >> 8));
        output.WriteByte((byte)adler);
        return output.ToArray();
    }

    private static uint Adler32(byte[] data)
    {
        const uint MOD_ADLER = 65521;
        uint a = 1, b = 0;
        foreach (var bt in data)
        {
            a = (a + bt) % MOD_ADLER;
            b = (b + a) % MOD_ADLER;
        }
        return (b << 16) | a;
    }

    private static byte[] Concat(params byte[][] arrays)
    {
        var result = new byte[arrays.Sum(a => a.Length)];
        var offset = 0;
        foreach (var a in arrays) { Buffer.BlockCopy(a, 0, result, offset, a.Length); offset += a.Length; }
        return result;
    }

    public sealed record AgoraTokenRequest(string? ChannelName, uint? Uid, string? Role, int? ExpireSeconds);
}
