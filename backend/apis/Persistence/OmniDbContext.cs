using Microsoft.EntityFrameworkCore;
using Omni.API.Models;

namespace Omni.API.Persistence;

public sealed class OmniDbContext : DbContext
{
    public OmniDbContext(DbContextOptions<OmniDbContext> options)
        : base(options)
    {
    }

    public DbSet<Camera> Cameras => Set<Camera>();
    public DbSet<PlateRecord> PlateRecords => Set<PlateRecord>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Camera>(entity =>
        {
            entity.ToTable("Cameras");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Id).HasMaxLength(64);
            entity.Property(e => e.Name).HasMaxLength(512);
            entity.Property(e => e.StreamUrl).HasMaxLength(4096);
            entity.Property(e => e.Status).HasMaxLength(32);
        });

        modelBuilder.Entity<PlateRecord>(entity =>
        {
            entity.ToTable("PlateRecords");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.PlateNumber).HasMaxLength(128);
            entity.Property(e => e.CameraId).HasMaxLength(128);
            entity.Property(e => e.ThumbnailPath).HasMaxLength(4096);
            entity.Property(e => e.FullFramePath).HasMaxLength(4096);
            entity.Property(e => e.VehicleType).HasMaxLength(64);
            entity.Property(e => e.Color).HasMaxLength(64);
            entity.Property(e => e.TrackId).HasColumnName("TrackingId").HasMaxLength(128);
            entity.Property(e => e.BoundingBox).HasMaxLength(8192);
            entity.Property(e => e.Direction).HasMaxLength(64);
        });
    }
}
