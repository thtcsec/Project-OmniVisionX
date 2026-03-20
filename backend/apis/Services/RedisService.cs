using StackExchange.Redis;

namespace Omni.API.Services;

public class RedisService
{
    private readonly IConnectionMultiplexer _redis;
    private readonly IDatabase _db;

    public RedisService(IConfiguration configuration)
    {
        var connectionString = configuration["Redis:ConnectionString"] ?? "localhost:6379";
        _redis = ConnectionMultiplexer.Connect(connectionString);
        _db = _redis.GetDatabase();
    }

    public IDatabase GetDatabase() => _db;

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
