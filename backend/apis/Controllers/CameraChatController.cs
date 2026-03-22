using System.Collections.Generic;
using System.Globalization;
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

        var difyKey = _env.Get("OMNI_DIFY_API_KEY");
        var difyBase = (_env.Get("OMNI_DIFY_BASE_URL") ?? "https://api.dify.ai/v1").TrimEnd('/');

        string provider;
        if (!string.IsNullOrWhiteSpace(difyKey))
            provider = "dify";
        else if (!string.IsNullOrWhiteSpace(openAiKey))
            provider = "openai";
        else if (!string.IsNullOrWhiteSpace(qwenKey))
            provider = "qwen";
        else
            provider = "";

        if (provider.Length == 0)
            return StatusCode(412, new { error = "No LLM configured. Set OMNI_DIFY_API_KEY, or OMNI_OPENAI_API_KEY, or OMNI_QWEN_API_KEY in Settings." });

        var systemPrompt = new StringBuilder();
        systemPrompt.Append("You are OmniVisionX, an assistant for a real-time traffic camera monitoring system. ");
        systemPrompt.Append("Answer briefly and concretely. ");
        systemPrompt.Append("If you reference detections, use the provided snapshot and state uncertainty when missing.");

        byte[]? frameJpeg = null;
        var wantVision = request.IncludeFrameImage != false;
        if (!string.IsNullOrWhiteSpace(request.CameraId) && wantVision)
            frameJpeg = await TryFetchSnapshotJpegAsync(request.CameraId.Trim(), ct);

        if (!string.IsNullOrWhiteSpace(request.CameraId))
        {
            var detectionText = await TryFetchDetectionContextAsync(request.CameraId.Trim(), ct);
            if (detectionText is not null)
            {
                systemPrompt.Append("\n\nStructured detection + OCR context (from pipeline; may be empty):\n");
                systemPrompt.Append(detectionText);
            }
        }

        var visionUsed = false;

        var exaKey = _env.Get("OMNI_EXA_API_KEY");
        var useExa = request.UseExaGrounding != false
            && !string.IsNullOrWhiteSpace(exaKey)
            && message.Length > 0;
        var exaUsed = false;
        if (useExa)
        {
            var grounding = await TryExaGroundingAsync(exaKey!, message, ct);
            if (grounding is not null)
            {
                exaUsed = true;
                systemPrompt.Append("\n\nWeb grounding (Exa search — verify against camera snapshot when they conflict):\n");
                systemPrompt.Append(grounding);
            }
        }

        if (provider == "dify")
        {
            if (frameJpeg is { Length: > 0 })
                systemPrompt.Append("\n\nNote: A live camera JPEG is shown to the user in the UI (not attached here). Use detection context above plus general knowledge.");
            var difyQuery = systemPrompt + "\n\nUser question: " + message;
            var reply = await CallDifyChatAsync(difyBase, difyKey!, difyQuery, ct);
            if (reply.Error != null)
                return StatusCode(reply.StatusCode ?? 502, new { error = reply.Error });
            return Ok(new ChatResponse("dify", "dify-chat", reply.Text ?? "", exaUsed, false));
        }

        var endpoint = provider == "openai"
            ? "https://api.openai.com/v1/chat/completions"
            : $"{qwenBase.TrimEnd('/')}/chat/completions";

        var model = provider == "openai" ? openAiModel : qwenModel;
        var apiKey = provider == "openai" ? openAiKey! : qwenKey!;

        using var client = _httpFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(90);
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

        object userMessage;
        if (frameJpeg is { Length: > 0 } && ShouldSendVision(provider, model))
        {
            var b64 = Convert.ToBase64String(frameJpeg);
            var dataUrl = "data:image/jpeg;base64," + b64;
            userMessage = new
            {
                role = "user",
                content = new object[]
                {
                    new { type = "text", text = message },
                    new { type = "image_url", image_url = new { url = dataUrl, detail = "low" } },
                },
            };
            visionUsed = true;
        }
        else
        {
            if (frameJpeg is { Length: > 0 })
                systemPrompt.Append("\n\nNote: A camera snapshot exists but this model is text-only; rely on structured detection lines above.");
            userMessage = new { role = "user", content = message };
        }

        var body = new
        {
            model,
            temperature = 0.2,
            messages = new object[]
            {
                new { role = "system", content = systemPrompt.ToString() },
                userMessage,
            },
        };

        var content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
        using var resp = await client.PostAsync(endpoint, content, ct);
        var text = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            return StatusCode((int)resp.StatusCode, new { error = text.Length > 500 ? text[..500] : text });

        var replyText = ExtractChatReply(text) ?? "";
        return Ok(new ChatResponse(provider, model, replyText, exaUsed, visionUsed));
    }

    private static bool ShouldSendVision(string provider, string model)
    {
        var m = model.ToLowerInvariant();
        if (provider == "openai")
        {
            if (m.Contains("gpt-3.5", StringComparison.OrdinalIgnoreCase))
                return false;
            // o1/o3 families are often text-only or different modalities — avoid vision unless using a known vision SKU.
            if (m.StartsWith("o1", StringComparison.OrdinalIgnoreCase) || m.StartsWith("o3", StringComparison.OrdinalIgnoreCase))
                return false;
            return m.Contains("gpt-4", StringComparison.OrdinalIgnoreCase)
                || m.Contains("gpt-5", StringComparison.OrdinalIgnoreCase);
        }

        if (provider == "qwen")
            return m.Contains("vl", StringComparison.OrdinalIgnoreCase) || m.Contains("vision", StringComparison.OrdinalIgnoreCase);

        return false;
    }

    private async Task<byte[]?> TryFetchSnapshotJpegAsync(string cameraId, CancellationToken ct)
    {
        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? _configuration["OmniObject__BaseUrl"] ?? "http://omni-object:8555").TrimEnd('/');
        var url = $"{baseUrl}/rtsp/snap/{Uri.EscapeDataString(cameraId)}";
        try
        {
            using var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(12);
            using var resp = await client.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
                return null;
            return await resp.Content.ReadAsByteArrayAsync(ct);
        }
        catch
        {
            return null;
        }
    }

    private async Task<string?> TryFetchDetectionContextAsync(string cameraId, CancellationToken ct)
    {
        var baseUrl = (_configuration["OmniObject:BaseUrl"] ?? _configuration["OmniObject__BaseUrl"] ?? "http://omni-object:8555").TrimEnd('/');
        var url = $"{baseUrl}/rtsp/detections/latest?cameraIds={Uri.EscapeDataString(cameraId)}";
        try
        {
            using var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(8);
            using var resp = await client.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
                return null;
            var json = await resp.Content.ReadAsStringAsync(ct);
            return FormatDetectionContext(json, cameraId);
        }
        catch
        {
            return null;
        }
    }

    private static string? FormatDetectionContext(string json, string cameraId)
    {
        if (string.IsNullOrWhiteSpace(json))
            return null;
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("items", out var items) || items.ValueKind != JsonValueKind.Object)
                return json.Length > 6000 ? json[..6000] + "…" : json;

            if (!items.TryGetProperty(cameraId, out var cam))
                return "No detection block for this camera yet (items may be empty).";

            var sb = new StringBuilder();
            if (cam.TryGetProperty("frame_width", out var fw) && cam.TryGetProperty("frame_height", out var fh)
                && fw.TryGetInt32(out var w) && fh.TryGetInt32(out var h))
                sb.AppendLine($"Frame size (inference space): {w}x{h} px");

            if (cam.TryGetProperty("timestamp", out var ts) && ts.TryGetDouble(out var tUnix))
                sb.AppendLine($"Snapshot unix time: {tUnix.ToString(CultureInfo.InvariantCulture)}");

            if (!cam.TryGetProperty("detections", out var dets) || dets.ValueKind != JsonValueKind.Array)
            {
                sb.AppendLine("No detections array.");
                return sb.ToString();
            }

            var n = 0;
            foreach (var d in dets.EnumerateArray())
            {
                n++;
                var cls = d.TryGetProperty("class_name", out var cn) && cn.ValueKind == JsonValueKind.String
                    ? cn.GetString() ?? "?"
                    : "?";
                var conf = 0.0;
                if (d.TryGetProperty("confidence", out var cf) && cf.TryGetDouble(out var cfd))
                    conf = cfd;
                var bboxStr = "";
                if (d.TryGetProperty("bbox", out var bb) && bb.ValueKind == JsonValueKind.Array)
                {
                    var parts = new List<string>();
                    foreach (var x in bb.EnumerateArray())
                    {
                        if (x.TryGetDouble(out var dv))
                            parts.Add(dv.ToString("0.##", CultureInfo.InvariantCulture));
                    }
                    bboxStr = string.Join(",", parts);
                }

                sb.AppendLine($"{n}. {cls} conf={conf:F2} bbox=[{bboxStr}] source={GetStr(d, "source")}");
                if (d.TryGetProperty("plate_text", out var pt) && pt.ValueKind == JsonValueKind.String)
                {
                    var p = pt.GetString();
                    if (!string.IsNullOrWhiteSpace(p))
                        sb.AppendLine($"   OCR / plate_text: {p}");
                }
            }

            if (n == 0)
                sb.AppendLine("(detections array empty)");

            var s = sb.ToString();
            return s.Length > 12000 ? s[..12000] + "…" : s;
        }
        catch
        {
            return json.Length > 6000 ? json[..6000] + "…" : json;
        }
    }

    private static string GetStr(JsonElement d, string name)
    {
        if (!d.TryGetProperty(name, out var el) || el.ValueKind != JsonValueKind.String)
            return "";
        return el.GetString() ?? "";
    }

    private async Task<string?> TryExaGroundingAsync(string apiKey, string userMessage, CancellationToken ct)
    {
        try
        {
            using var client = _httpFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(25);

            var payload = new
            {
                query = userMessage,
                type = "auto",
                numResults = 4,
                contents = new { text = true },
            };
            var json = JsonSerializer.Serialize(payload);
            using var req = new HttpRequestMessage(HttpMethod.Post, "https://api.exa.ai/search")
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json"),
            };
            req.Headers.TryAddWithoutValidation("x-api-key", apiKey);

            using var resp = await client.SendAsync(req, ct);
            var raw = await resp.Content.ReadAsStringAsync(ct);
            if (!resp.IsSuccessStatusCode)
                return null;

            return SummarizeExaSearchResults(raw);
        }
        catch
        {
            return null;
        }
    }

    private static string? SummarizeExaSearchResults(string rawJson)
    {
        try
        {
            using var doc = JsonDocument.Parse(rawJson);
            if (!doc.RootElement.TryGetProperty("results", out var results) || results.ValueKind != JsonValueKind.Array)
                return null;
            var sb = new StringBuilder();
            var i = 0;
            foreach (var r in results.EnumerateArray())
            {
                if (i >= 4) break;
                var title = r.TryGetProperty("title", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString() : "";
                var url = r.TryGetProperty("url", out var u) && u.ValueKind == JsonValueKind.String ? u.GetString() : "";
                string? snippet = null;
                if (r.TryGetProperty("text", out var tx) && tx.ValueKind == JsonValueKind.String)
                    snippet = tx.GetString();
                else if (r.TryGetProperty("highlights", out var hi) && hi.ValueKind == JsonValueKind.Array && hi.GetArrayLength() > 0)
                    snippet = hi[0].GetString();

                if (!string.IsNullOrWhiteSpace(title) || !string.IsNullOrWhiteSpace(snippet))
                {
                    sb.Append(i + 1).Append(". ");
                    if (!string.IsNullOrWhiteSpace(title))
                        sb.Append(title).Append(" — ");
                    if (!string.IsNullOrWhiteSpace(url))
                        sb.Append(url).Append('\n');
                    if (!string.IsNullOrWhiteSpace(snippet))
                    {
                        var s = snippet.Length > 400 ? snippet[..400] + "…" : snippet;
                        sb.Append(s).Append("\n\n");
                    }
                }
                i++;
            }
            var s2 = sb.ToString().Trim();
            return s2.Length > 0 ? s2 : null;
        }
        catch
        {
            return null;
        }
    }

    private static async Task<(string? Text, string? Error, int? StatusCode)> CallDifyChatAsync(
        string baseV1, string apiKey, string query, CancellationToken ct)
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(90) };
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);

            var body = new
            {
                inputs = new Dictionary<string, object>(),
                query,
                response_mode = "blocking",
                user = "omnivision-chat",
            };
            var json = JsonSerializer.Serialize(body);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");
            using var resp = await client.PostAsync($"{baseV1}/chat-messages", content, ct);
            var text = await resp.Content.ReadAsStringAsync(ct);
            if (!resp.IsSuccessStatusCode)
                return (null, text.Length > 600 ? text[..600] : text, (int)resp.StatusCode);

            using var doc = JsonDocument.Parse(text);
            var root = doc.RootElement;
            if (root.TryGetProperty("answer", out var ans) && ans.ValueKind == JsonValueKind.String)
                return (ans.GetString(), null, null);
            return (null, "Dify response missing answer field", 502);
        }
        catch (Exception ex)
        {
            return (null, ex.Message, 502);
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

        /// <summary>When false, skip Exa web search even if OMNI_EXA_API_KEY is set.</summary>
        [JsonPropertyName("useExaGrounding")]
        public bool? UseExaGrounding { get; set; }

        /// <summary>When false, skip fetching JPEG from omni-object (text-only chat).</summary>
        [JsonPropertyName("includeFrameImage")]
        public bool? IncludeFrameImage { get; set; }
    }

    public sealed record ChatResponse(string Provider, string Model, string Reply, bool ExaUsed = false, bool VisionUsed = false);
}
