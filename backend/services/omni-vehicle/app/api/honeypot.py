"""
Security Honeypot endpoints (Chương 4 — Phase 2 Hardened).
Migrated from root ai-engine/app/main.py to omni-vehicle.

Phase 2 upgrades:
- Expanded trap surface (exec, cmd, debug, phpinfo, phpmyadmin, wp-admin,
  wp-login, .env, .git, actuator, console, manage.py, eval)
- In-memory repeat-offender tracking with escalating severity
- Logs to both SystemLogs AND dedicated SecurityLogs table
- Fake delayed response to slow down scanners
"""
import asyncio
import logging
import os
import time
from collections import defaultdict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("omni-vehicle.honeypot")

router = APIRouter(tags=["Security Honeypot"])

# Web CMS Internal URLs (Docker network)
WEB_API_URL = os.getenv("WEB_API_URL", "http://web-cms:5000/api/SystemLogs")
# The Web CMS doesn't have a /api/SecurityLogs endpoint, only SystemLogs for broadcasting

# In-memory offender tracking: ip → {"count": int, "first_seen": float, "last_seen": float}
_offenders: dict[str, dict] = defaultdict(lambda: {"count": 0, "first_seen": 0.0, "last_seen": 0.0})
# Max offenders to track (prevent memory abuse)
_MAX_OFFENDERS = 10_000
# Lock for thread-safe offender tracking in async context
_offenders_lock = asyncio.Lock()


def _get_severity(count: int) -> str:
    """Escalate severity based on repeat offenses.
    Maps to Omni.API.Models.LogLevel enum: Info, Warning, Error, Critical
    """
    if count >= 5:
        return "Critical"
    elif count >= 2:
        return "Error"
    return "Error"  # Any honeypot hit is a security incident


async def log_security_event(ip: str, endpoint: str, method: str, hit_count: int):
    """
    Send security event to Web CMS API via HTTP.
    Logs to both SystemLogs (SignalR broadcast) AND SecurityLogs (dedicated table).
    """
    try:
        import httpx

        severity = _get_severity(hit_count)
        payload = {
            "Level": severity,
            "Source": "AI_Honeypot",
            "Message": f"HONEYPOT TRIGGERED: {endpoint}",
            "Details": (
                f"Intruder IP: {ip} | Method: {method} | "
                f"Hit #{hit_count} | Severity: {severity} | Action: Blocked"
            ),
        }

        internal_secret = os.getenv("INTERNAL_SECRET", "")
        headers = {"X-Internal-Secret": internal_secret}

        async with httpx.AsyncClient() as client:
            # Fire both log endpoints in parallel
            tasks = [
                client.post(WEB_API_URL, json=payload, headers=headers, timeout=5.0),
            ]


            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Security log send error: %s", r)
                elif hasattr(r, "status_code") and r.status_code >= 400:
                    logger.warning("Security log HTTP error: %s", r.status_code)

    except Exception as e:
        logger.warning("Error sending security log: %s", e)


# --- Expanded Trap Routes ---
# All common scanner targets

@router.get("/api/admin/shell")
@router.post("/api/admin/shell")
@router.get("/api/admin/exec")
@router.post("/api/admin/exec")
@router.get("/api/admin/cmd")
@router.post("/api/admin/cmd")
@router.get("/api/admin/debug")
@router.get("/api/admin/phpinfo")
@router.get("/api/admin/phpmyadmin")
@router.get("/api/admin/wp-admin")
@router.get("/api/admin/wp-login")
@router.get("/api/system/config")
@router.post("/api/upload/php")
@router.get("/.env")
@router.get("/.git/config")
@router.get("/actuator")
@router.get("/actuator/health")
@router.get("/console")
@router.get("/manage.py")
@router.post("/eval")
@router.get("/api/admin/eval")
async def honeypot(request: Request):
    """
    Fake Administrative Endpoint (Honeypot).
    Logs suspicious access attempts with repeat-offender escalation.
    Adds artificial delay to slow down automated scanners.
    """
    client_ip = request.client.host or "unknown"
    now = time.time()

    async with _offenders_lock:
        # Evict stale entries if at capacity
        if len(_offenders) >= _MAX_OFFENDERS and client_ip not in _offenders:
            stale_cutoff = now - 86400  # 24 hours
            stale_ips = [ip for ip, v in list(_offenders.items()) if v["last_seen"] < stale_cutoff]
            for ip in stale_ips:
                _offenders.pop(ip, None)

        # Track offender
        if len(_offenders) < _MAX_OFFENDERS or client_ip in _offenders:
            entry = _offenders[client_ip]
            entry["count"] += 1
            if entry["first_seen"] == 0.0:
                entry["first_seen"] = now
            entry["last_seen"] = now
            hit_count = entry["count"]
        else:
            hit_count = 1

    severity = _get_severity(hit_count)
    logger.critical(
        "🚨 [SECURITY ALERT] HONEYPOT TRIGGERED! IP: %s | Endpoint: %s | Hit #%d | Severity: %s",
        client_ip, request.url.path, hit_count, severity,
    )

    await log_security_event(client_ip, request.url.path, request.method, hit_count)

    # Artificial delay to slow down scanners (500ms-2s based on severity)
    delay = min(0.5 * hit_count, 2.0)
    await asyncio.sleep(delay)

    return JSONResponse(
        status_code=403,
        content={
            "error": "Access Denied",
            "code": "SYS_ROOT_PROTECTED",
            "message": "Your IP has been logged and reported to the administrator.",
        },
    )


@router.get("/api/honeypot/stats")
async def honeypot_stats(request: Request):
    """Internal endpoint: view honeypot hit statistics (for admin dashboard)."""
    expected_secret = os.getenv("INTERNAL_SECRET", "")
    if expected_secret and request.headers.get("X-Internal-Secret") != expected_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    # Only return top 20 offenders sorted by count
    top = sorted(_offenders.items(), key=lambda x: x[1]["count"], reverse=True)[:20]
    return {
        "total_tracked_ips": len(_offenders),
        "top_offenders": [
            {
                "ip": ip,
                "count": data["count"],
                "first_seen": data["first_seen"],
                "last_seen": data["last_seen"],
                "severity": _get_severity(data["count"]),
            }
            for ip, data in top
        ],
    }

