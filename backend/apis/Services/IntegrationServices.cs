namespace Omni.API.Services;

// ElevenLabs TTS Integration
public interface IElevenLabsService
{
    Task<string> SynthesizeSpeechAsync(string text, string voiceId = "21m00Tcm4TlvDq8ikWAM");
    Task<IEnumerable<Voice>> GetVoicesAsync();
}

public record Voice(string Id, string Name, Category Category);
public enum Category { Cloned, Professional, Raw }

// Agora Video Integration
public interface IAgoraService
{
    Task<string> GenerateTokenAsync(string channelName, string uid, int privilegeExpiredTs = 3600);
    Task CreateChannelAsync(string channelName);
    Task DeleteChannelAsync(string channelName);
}

// OpenAI Integration
public interface IOpenAIService
{
    Task<string> TranscribeAudioAsync(byte[] audioData);
    Task<string> AnalyzeImageAsync(string imageUrl, string prompt);
    Task<string> GenerateSummaryAsync(string text);
}

// Sponsor/Advertiser Integration
public interface ISponsorService
{
    Task<IEnumerable<Campaign>> GetActiveCampaignsAsync();
    Task RecordImpressionAsync(string campaignId, string cameraId);
    Task RecordClickAsync(string campaignId, string cameraId);
}

public record Campaign(string Id, string Name, string Content, int Priority, DateTime StartDate, DateTime EndDate);
