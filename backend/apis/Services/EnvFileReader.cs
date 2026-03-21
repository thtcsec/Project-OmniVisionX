namespace Omni.API.Services;

/// <summary>
/// Reads key=value pairs from a .env file on disk.
/// Used so that values saved via the Settings UI (written to /app/.env)
/// are immediately visible without restarting the container.
///
/// Priority order (highest first):
///   1. Process environment variable (injected by docker-compose at startup)
///   2. .env file on disk (written at runtime via Settings UI)
///   3. IConfiguration (appsettings.json / env override)
/// </summary>
public sealed class EnvFileReader
{
    private readonly IConfiguration _configuration;
    private readonly string _envFilePath;

    public EnvFileReader(IConfiguration configuration)
    {
        _configuration = configuration;
        _envFilePath = configuration["EnvEditor:EnvFilePath"]
                    ?? configuration["EnvEditor__EnvFilePath"]
                    ?? "/app/.env";
    }

    /// <summary>
    /// Get a value by trying multiple key names in order.
    /// Reads process env first, then .env file, then IConfiguration.
    /// </summary>
    public string? Get(params string[] keys)
    {
        // 1. Process environment (injected at container start by docker-compose)
        foreach (var k in keys)
        {
            var v = Environment.GetEnvironmentVariable(k);
            if (!string.IsNullOrWhiteSpace(v)) return v;
        }

        // 2. .env file on disk (written by Settings UI at runtime)
        var fileVars = ReadEnvFileCached();
        foreach (var k in keys)
        {
            if (fileVars.TryGetValue(k, out var v) && !string.IsNullOrWhiteSpace(v))
                return v;
        }

        // 3. IConfiguration (appsettings.json or compose env with dotted keys)
        foreach (var k in keys)
        {
            var v = _configuration[k];
            if (!string.IsNullOrWhiteSpace(v)) return v;
        }

        return null;
    }

    public bool Has(params string[] keys) => !string.IsNullOrWhiteSpace(Get(keys));

    // ── .env file cache (5-second TTL to avoid hammering disk) ──────────
    private Dictionary<string, string> _fileCache = new(StringComparer.OrdinalIgnoreCase);
    private DateTime _cacheExpiry = DateTime.MinValue;
    private readonly object _cacheLock = new();

    private Dictionary<string, string> ReadEnvFileCached()
    {
        lock (_cacheLock)
        {
            if (DateTime.UtcNow < _cacheExpiry)
                return _fileCache;

            _fileCache = ReadEnvFile(_envFilePath);
            _cacheExpiry = DateTime.UtcNow.AddSeconds(5);
            return _fileCache;
        }
    }

    private static Dictionary<string, string> ReadEnvFile(string path)
    {
        var vars = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (!File.Exists(path)) return vars;
        try
        {
            foreach (var raw in File.ReadAllLines(path))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                if (line.StartsWith("export ", StringComparison.OrdinalIgnoreCase))
                    line = line[7..].TrimStart();
                var idx = line.IndexOf('=');
                if (idx <= 0) continue;
                var key = line[..idx].Trim();
                var val = Unquote(line[(idx + 1)..].Trim());
                vars[key] = val;
            }
        }
        catch { /* ignore IO errors — file may be locked during write */ }
        return vars;
    }

    private static string Unquote(string value)
    {
        if (value.Length >= 2 &&
            ((value[0] == '"' && value[^1] == '"') || (value[0] == '\'' && value[^1] == '\'')))
            return value[1..^1].Replace("\\\"", "\"", StringComparison.Ordinal);
        return value;
    }
}
