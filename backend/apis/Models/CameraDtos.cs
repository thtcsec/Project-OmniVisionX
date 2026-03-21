using System.ComponentModel.DataAnnotations;

namespace Omni.API.Models;

/// <summary>Payload for POST/PUT — hackathon: no auth.</summary>
public class CameraWriteDto
{
    [Required, MaxLength(255)]
    public string Name { get; set; } = "";

    [MaxLength(500)]
    public string StreamUrl { get; set; } = "";

    [RegularExpression("^(online|offline)$")]
    public string Status { get; set; } = "offline";

    public bool EnableObjectDetection { get; set; }
    public bool EnablePlateOcr { get; set; }
    public bool EnableFaceRecognition { get; set; }
}
