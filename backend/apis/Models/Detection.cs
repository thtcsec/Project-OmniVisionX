namespace Omni.API.Models;

public class Detection
{
    public int TrackId { get; set; }
    public string ClassName { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public int[] Bbox { get; set; } = Array.Empty<int>(); // [x1, y1, x2, y2]
    public string CameraId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

public class PlateEvent
{
    public string PlateText { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public int[] Bbox { get; set; } = Array.Empty<int>();
    public string VehicleType { get; set; } = "unknown";
    public int? TrackId { get; set; }
    public string CameraId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

public class FaceEvent
{
    public int TrackId { get; set; }
    public float Confidence { get; set; }
    public int[] Bbox { get; set; } = Array.Empty<int>();
    public float[] Embedding { get; set; } = Array.Empty<float>();
    public string? MatchedPersonId { get; set; }
    public string CameraId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

public class Camera
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string StreamUrl { get; set; } = string.Empty;
    public string Status { get; set; } = "offline";
    public bool EnableObjectDetection { get; set; }
    public bool EnablePlateOcr { get; set; }
    public bool EnableFaceRecognition { get; set; }
}

public class CameraStats
{
    public string CameraId { get; set; } = string.Empty;
    public bool Healthy { get; set; }
    public int TotalDetections { get; set; }
    public int TotalPlates { get; set; }
    public DateTime LastFrameTime { get; set; }
}
