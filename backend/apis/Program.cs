using Microsoft.EntityFrameworkCore;
using Omni.API.Data;
using Omni.API.Hubs;
using Omni.API.Services;

var builder = WebApplication.CreateBuilder(args);

// Add MVC Controllers
builder.Services.AddControllers();

// Add SignalR
builder.Services.AddSignalR();

// Add PostgreSQL with pgvector
builder.Services.AddDbContext<OmniDbContext>(options =>
    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

// Add Redis
builder.Services.AddSingleton<RedisService>();

// Shared .env file reader — used by integration controllers to pick up
// values saved at runtime via the Settings UI without container restart.
builder.Services.AddSingleton<EnvFileReader>();

// HTTP client for proxying to omni-object (live detection overlays)
builder.Services.AddHttpClient();

// MediaMTX — register camera paths for browser HLS/WebRTC (path name = camera Id)
builder.Services.AddHttpClient<MediaMtxPathRegistrar>(client =>
{
    client.Timeout = TimeSpan.FromSeconds(12);
});
builder.Services.AddHostedService<MediaMtxStartupSync>();

// Add Background Services
builder.Services.AddHostedService<OmniEventConsumerService>();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

// Add OpenAPI/Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();
app.UseRouting();

app.MapControllers();
app.MapHub<OmniHub>("/hubs/omni");

app.MapGet("/health", () => Results.Ok(new { status = "ok", service = "OmniAPI", version = "1.0.0" }));

// Ensure database created
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<OmniDbContext>();
    db.Database.EnsureCreated();
}

app.Run();
