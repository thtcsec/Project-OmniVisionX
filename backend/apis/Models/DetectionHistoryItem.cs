namespace Omni.API.Models;

/// <summary>JSON shape matches frontend <c>Detection</c> (Camera detail history table).</summary>
public sealed class DetectionHistoryItem
{
    public string Id { get; set; } = "";
    public string CameraId { get; set; } = "";
    /// <summary>Display label: YOLO class or vehicle + plate.</summary>
    public string Type { get; set; } = "";
    public float Confidence { get; set; }
    public string Timestamp { get; set; } = "";
    public int? TrackId { get; set; }
    public BboxNormDto? Bbox { get; set; }
    /// <summary>detections | vehicles</summary>
    public string? Source { get; set; }
    public Dictionary<string, string>? Metadata { get; set; }
}

public sealed class BboxNormDto
{
    public int X1 { get; set; }
    public int Y1 { get; set; }
    public int X2 { get; set; }
    public int Y2 { get; set; }
}
