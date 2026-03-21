namespace Omni.API.Models;

public sealed class PlateRecord
{
    public Guid Id { get; set; }
    public string PlateNumber { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string CameraId { get; set; } = string.Empty;
    public string? ThumbnailPath { get; set; }
    public string? FullFramePath { get; set; }
    public string VehicleType { get; set; } = "unknown";
    public float Confidence { get; set; }
    public string? Color { get; set; }
    public string? TrackId { get; set; }
    public string? BoundingBox { get; set; }
    public string? Direction { get; set; }
    public bool IsDeleted { get; set; }
}

