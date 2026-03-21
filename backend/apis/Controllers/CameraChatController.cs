using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/integrations/chat")]
public sealed class CameraChatController : ControllerBase
{
    private readonly IHttpClientFactory _httpFactory;
    private readonly IConfiguration _configuration;
    private readonly EnvFileReader _env;

    public CameraChatController(IHttpClientFactory httpFactory, IConfiguration configuration, EnvFileReader env)
    {
        _httpFactory = httpFactory;
        _configuration = configuration;
        _env = env;
    }

    [HttpPost]
    public async Task<ActionResult<ChatResponse>> Chat([FromBody] ChatRequest request, CancellationToken ct)
    {
        var message = (request.Message ?? "").Trim();
        if (message.Length == 0)
            return BadRequest(new { error = "Missing message" });

        var openAiKey = _env.Get("OMNI_OPENAI_API_KEY");
        var openAiModel = _env.Get("OMNI_OPENAI_MODEL_ID") ?? "gpt-4o-mini";

        var qwenKey = _env.Get("OMNI_QWEN_API_KEY");
        var qwenBase = _env.Get("OMNI_QWEN_BASE_URL") ?? "https://dashscope.aliyuncs.com/compatible-mode/v1";
        var qwenModel = _env.Get("OMNI_QWEN_MODEL_ID") ?? "qwen-plus";

        var provider = !string.IsNullOrWhiteSpace(openAiKey) ? "openai" : (!string.IsNullOrWhiteSpace(qwenKey) ? "qwen" : "");
        if (provider.Length == 0)
            return StatusCode(412, new { error = "No LLM configured. Set OMNI_OPENAI_API_KEY or OMNI_QWEN_API_KEY in Settings." });

        var systemPrompt = new StringBuilder();
        systemPrompt.Append("You are OmniVisionX, an assistant for a real-time traffic camera monitoring system. ");
        systemPrompt.Append("Answer briefly and concretely. ");
        systemPrompt.Append("If you reference detections, use the provided snapshot and state uncertainty when missing.");

        if (!string.IsNullOrWhiteSpace(request.CameraId))
        {
            var snapshot = await TryFetchLatestDetectionsAsync(request.CameraId.Trim(), ct);
            if (snapshot is not null)
            {
                systemPrompt.Append("\n\nLatest detections snapshot (may be empty):\n");
                systemPrompt.Append(snapshot);
            }
        }

        var endpoint = provider == "openai"
            ? "https://api.openai.com/v1/chat/completions"
            : $"{qwenBase.TrimEnd('/')}/chat/completions";

        var model = provider == "openai" ? openAiModel : qwenModel;
        var apiKey = provider == "openai" ? openAiKey! : qwenKey!;

        using var client = _httpFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(30);
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        var body = new
        {
            model,
            temperature = 0.2,
            messages = new object[]
            {
                new { role = "system", content = systemPrompt.ToString() },
                new { role = "user", content = message },
            },
        };

        var content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
        using var resp = await client.PostAsync(endpoint, content, ct);
        var text = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            return StatusCode((int)resp.StatusCode, new { error = text.Length > 500 ? text[..500] : text });

        var reply = ExtractChatReply(text) ?? "";
        return Ok(new ChatResponse(provider, model, reply));
    }

    private async Task<string?> TryFetchLatestDetectionsAsync(string cameraId, CancellationToken ct)
    {
        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? _configuration["OmniObject__BaseUrl"] ?? "http://omni-object:8555").TrimEnd('/');
        var url = $"{baseUrl}/rtsp/detections/latest?cameraIds={Uri.EscapeDataString(cameraId)}";
        try
        {
            using var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(6);
            using var resp = await client.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode) return null;
            var json = await resp.Content.ReadAsStringAsync(ct);
            return json.Length > 800 ? json[..800] : json;
        }
        catch
        {
            return null;
        }
    }

    private static string? ExtractChatReply(string rawJson)
    {
        try
        {
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;
            if (root.TryGetProperty("choices", out var choices) && choices.ValueKind == JsonValueKind.Array && choices.GetArrayLength() > 0)
            {
                var first = choices[0];
                if (first.TryGetProperty("message", out var msg) && msg.ValueKind == JsonValueKind.Object)
                {
                    if (msg.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String)
                        return c.GetString();
                }
                if (first.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
                    return t.GetString();
            }
        }
        catch { }
        return null;
    }

    public sealed class ChatRequest
    {
        [JsonPropertyName("message")]
        public string? Message { get; set; }

        [JsonPropertyName("cameraId")]
        public string? CameraId { get; set; }
    }
    public sealed record ChatResponse(string Provider, string Model, string Reply);
}
