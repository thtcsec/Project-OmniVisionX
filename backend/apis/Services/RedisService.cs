using StackExchange.Redis;

namespace Omni.API.Services;

public class RedisService
{
    private readonly IConnectionMultiplexer _redis;
    private readonly IDatabase _db;

    public RedisService(IConfiguration configuration)
    {
        var connectionString = configuration["Redis:ConnectionString"] ?? "localhost:6379";
        var opts = ConfigurationOptions.Parse(connectionString);
        opts.AbortOnConnectFail = false;
        opts.ConnectRetry = 5;
        opts.ConnectTimeout = 10_000;
        _redis = ConnectionMultiplexer.Connect(opts);
        _db = _redis.GetDatabase();
    }

    public IDatabase GetDatabase() => _db;

    /// <summary>Newest-first slice of a Redis stream (XREVRANGE … COUNT).</summary>
    public async Task<StreamEntry[]> StreamRevRangeAsync(RedisKey key, int count)
    {
        if (count <= 0)
            return Array.Empty<StreamEntry>();
        try
        {
            var len = await _db.StreamLengthAsync(key);
            if (len == 0)
                return Array.Empty<StreamEntry>();
            return await _db.StreamRangeAsync(key, "-", "+", count, Order.Descending);
        }
        catch (RedisException)
        {
            return Array.Empty<StreamEntry>();
        }
    }

    public async Task<StreamEntry[]> StreamReadAsync(string key, string group, int count = 100)
    {
        var entries = await _db.StreamReadGroupAsync(key, group, "consumer", count);
        return entries;
    }

    public async Task StreamAckAsync(string key, string group, RedisValue messageId)
    {
        await _db.StreamAcknowledgeAsync(key, group, messageId);
    }

    public async Task<long> StreamLengthAsync(string key)
    {
        return await _db.StreamLengthAsync(key);
    }
}
