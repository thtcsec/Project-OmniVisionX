using Microsoft.AspNetCore.Mvc;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/settings/integrations/env")]
public sealed class IntegrationsEnvController : ControllerBase
{
    private static readonly EnvVarSpec[] Specs =
    [
        new("Agora", "App ID",          "OMNI_AGORA_APP_ID",          false),
        new("Agora", "App Certificate", "OMNI_AGORA_APP_CERTIFICATE",  true),
        new("ElevenLabs", "API Key",    "OMNI_ELEVENLABS_API_KEY",     true),
        new("ElevenLabs", "Voice ID",   "OMNI_ELEVENLABS_VOICE_ID",    false),
        new("ElevenLabs", "Model ID",   "OMNI_ELEVENLABS_MODEL_ID",    false),
        new("OpenAI", "API Key",        "OMNI_OPENAI_API_KEY",         true),
        new("OpenAI", "Model ID",       "OMNI_OPENAI_MODEL_ID",        false),
        new("Qwen", "API Key",          "OMNI_QWEN_API_KEY",           true),
        new("Qwen", "Base URL",         "OMNI_QWEN_BASE_URL",          false),
        new("Qwen", "Model ID",         "OMNI_QWEN_MODEL_ID",          false),
    ];

    private readonly IConfiguration _configuration;

    public IntegrationsEnvController(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    private bool Enabled
    {
        get
        {
            // Check env var first (docker-compose passes OMNI_ENV_EDITOR_ENABLED=true)
            var fromEnv = Environment.GetEnvironmentVariable("OMNI_ENV_EDITOR_ENABLED");
            if (!string.IsNullOrWhiteSpace(fromEnv))
                return string.Equals(fromEnv, "true", StringComparison.OrdinalIgnoreCase);
            // Fallback to appsettings / compose-injected config key
            var fromCfg = _configuration["EnvEditor:Enabled"] ?? _configuration["EnvEditor__Enabled"];
            return string.Equals(fromCfg, "true", StringComparison.OrdinalIgnoreCase);
        }
    }

    private string EnvFilePath =>
        _configuration["EnvEditor:EnvFilePath"]
        ?? _configuration["EnvEditor__EnvFilePath"]
        ?? "/app/.env";

    [HttpGet]
    public async Task<ActionResult<IReadOnlyList<EnvVarDto>>> Get(CancellationToken ct)
    {
        if (!Enabled)
            return NotFound();

        var fileVars = await ReadEnvFileAsync(EnvFilePath, ct);

        var result = Specs.Select(spec =>
        {
            var value = fileVars.TryGetValue(spec.Key, out var fromFile) ? fromFile : Environment.GetEnvironmentVariable(spec.Key);
            var isSet = !string.IsNullOrWhiteSpace(value);
            return new EnvVarDto(
                Group: spec.Group,
                Label: spec.Label,
                Key: spec.Key,
                IsSecret: spec.IsSecret,
                IsSet: isSet,
                Value: spec.IsSecret ? null : (value ?? "")
            );
        }).ToArray();

        return Ok(result);
    }

    [HttpPut]
    public async Task<IActionResult> Put([FromBody] EnvUpdateRequest request, CancellationToken ct)
    {
        if (!Enabled)
            return NotFound();

        if (request is not { Updates.Count: > 0 })
            return BadRequest(new { error = "No updates provided" });

        var allowed = new HashSet<string>(Specs.Select(s => s.Key), StringComparer.OrdinalIgnoreCase);
        foreach (var u in request.Updates)
        {
            if (string.IsNullOrWhiteSpace(u.Key) || !allowed.Contains(u.Key))
                return BadRequest(new { error = $"Unsupported key: {u.Key}" });
        }

        var updates = request.Updates
            .Where(u => !string.IsNullOrWhiteSpace(u.Key))
            .ToDictionary(u => u.Key.Trim(), u => u.Value ?? "", StringComparer.OrdinalIgnoreCase);

        await WriteEnvFileAsync(EnvFilePath, updates, ct);

        foreach (var (k, v) in updates)
            Environment.SetEnvironmentVariable(k, string.IsNullOrWhiteSpace(v) ? null : v);

        return NoContent();
    }

    private static async Task<Dictionary<string, string>> ReadEnvFileAsync(string path, CancellationToken ct)
    {
        var vars = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (!System.IO.File.Exists(path))
            return vars;

        var lines = await System.IO.File.ReadAllLinesAsync(path, ct);
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith("#"))
                continue;

            if (line.StartsWith("export ", StringComparison.OrdinalIgnoreCase))
                line = line[7..].TrimStart();

            var idx = line.IndexOf('=');
            if (idx <= 0)
                continue;

            var key = line[..idx].Trim();
            var value = line[(idx + 1)..].Trim();
            vars[key] = Unquote(value);
        }

        return vars;
    }

    private static async Task WriteEnvFileAsync(string path, IReadOnlyDictionary<string, string> updates, CancellationToken ct)
    {
        var lines = new List<string>();
        if (System.IO.File.Exists(path))
            lines.AddRange(await System.IO.File.ReadAllLinesAsync(path, ct));

        var indexByKey = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < lines.Count; i++)
        {
            var raw = lines[i].TrimStart();
            if (raw.StartsWith("#"))
                continue;

            var probe = raw.StartsWith("export ", StringComparison.OrdinalIgnoreCase) ? raw[7..].TrimStart() : raw;
            var idx = probe.IndexOf('=');
            if (idx <= 0)
                continue;

            var key = probe[..idx].Trim();
            if (!indexByKey.ContainsKey(key))
                indexByKey[key] = i;
        }

        foreach (var (key, value) in updates)
        {
            var normalizedKey = key.Trim();
            if (string.IsNullOrEmpty(normalizedKey))
                continue;

            if (string.IsNullOrWhiteSpace(value))
            {
                if (indexByKey.TryGetValue(normalizedKey, out var idx))
                    lines[idx] = "";
                continue;
            }

            var rendered = $"{normalizedKey}={QuoteIfNeeded(value)}";
            if (indexByKey.TryGetValue(normalizedKey, out var existingIdx))
                lines[existingIdx] = rendered;
            else
                lines.Add(rendered);
        }

        var finalLines = lines.Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? "/");
        await System.IO.File.WriteAllLinesAsync(path, finalLines, ct);
    }

    private static string Unquote(string value)
    {
        if (value.Length >= 2 && ((value.StartsWith('"') && value.EndsWith('"')) || (value.StartsWith('\'') && value.EndsWith('\''))))
            return value[1..^1].Replace("\\\"", "\"", StringComparison.Ordinal);
        return value;
    }

    private static string QuoteIfNeeded(string value)
    {
        var needs = value.Any(char.IsWhiteSpace) || value.Contains('"') || value.Contains('#') || value.Contains('=');
        if (!needs)
            return value;
        return $"\"{value.Replace("\"", "\\\"", StringComparison.Ordinal)}\"";
    }

    private sealed record EnvVarSpec(string Group, string Label, string Key, bool IsSecret);
    public sealed record EnvVarDto(string Group, string Label, string Key, bool IsSecret, bool IsSet, string? Value);
    public sealed record EnvUpdateRequest(List<EnvUpdateItem> Updates);
    public sealed record EnvUpdateItem(string Key, string? Value);
}

