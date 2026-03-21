using Microsoft.AspNetCore.Mvc;
using Omni.API.Services;

namespace Omni.API.Controllers;

[ApiController]
[Route("api/integrations/status")]
public sealed class IntegrationsStatusController : ControllerBase
{
    private readonly EnvFileReader _env;

    public IntegrationsStatusController(EnvFileReader env)
    {
        _env = env;
    }

    [HttpGet]
    public IActionResult Get()
    {
        var status = new
        {
            agora = new { configured = _env.Has("OMNI_AGORA_APP_ID") && _env.Has("OMNI_AGORA_APP_CERTIFICATE") },
            elevenlabs = new { configured = _env.Has("OMNI_ELEVENLABS_API_KEY") },
            valsea = new { configured = _env.Has("OMNI_VALSEA_API_KEY") },
            openai = new { configured = _env.Has("OMNI_OPENAI_API_KEY") },
            exa = new { configured = _env.Has("OMNI_EXA_API_KEY") },
            qwen = new { configured = _env.Has("OMNI_QWEN_API_KEY") },
            dify = new { configured = _env.Has("OMNI_DIFY_API_KEY") },
        };

        return Ok(status);
    }
}
