using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/integrations/tts")]
public sealed class IntegrationsTtsController : ControllerBase
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly EnvFileReader _env;

    public IntegrationsTtsController(IHttpClientFactory httpFactory, EnvFileReader env)
    {
        _httpFactory = httpFactory;
        _env = env;
    }

    [HttpPost]
    public async Task<IActionResult> Speak([FromBody] TtsRequest request, CancellationToken ct)
    {
        var text = (request.Text ?? "").Trim();
        if (text.Length == 0)
            return BadRequest(new { error = "Missing text" });

        var apiKey = _env.Get("OMNI_ELEVENLABS_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
            return StatusCode(412, new { error = "ElevenLabs not configured. Set OMNI_ELEVENLABS_API_KEY in Settings." });

        var voiceId = request.VoiceId?.Trim();
        if (string.IsNullOrWhiteSpace(voiceId))
            voiceId = _env.Get("OMNI_ELEVENLABS_VOICE_ID") ?? "21m00Tcm4TlvDq8ikWAM";

        var modelId = request.ModelId?.Trim();
        if (string.IsNullOrWhiteSpace(modelId))
            modelId = _env.Get("OMNI_ELEVENLABS_MODEL_ID") ?? "eleven_multilingual_v2";

        using var client = _httpFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(30);
        client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("audio/mpeg"));
        client.DefaultRequestHeaders.Add("xi-api-key", apiKey);

        var body = new
        {
            text,
            model_id = modelId,
            voice_settings = new { stability = 0.45, similarity_boost = 0.75 },
        };

        var json = JsonSerializer.Serialize(body);
        using var resp = await client.PostAsync(
            $"https://api.elevenlabs.io/v1/text-to-speech/{Uri.EscapeDataString(voiceId)}",
            new StringContent(json, Encoding.UTF8, "application/json"),
            ct
        );

        var bytes = await resp.Content.ReadAsByteArrayAsync(ct);
        if (!resp.IsSuccessStatusCode)
        {
            var msg = bytes.Length > 0 ? Encoding.UTF8.GetString(bytes) : (resp.ReasonPhrase ?? "");
            return StatusCode((int)resp.StatusCode, new { error = msg.Length > 500 ? msg[..500] : msg });
        }

        return File(bytes, "audio/mpeg");
    }

    public sealed record TtsRequest(string? Text, string? VoiceId, string? ModelId);
}
