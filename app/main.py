"""FastAPI app factory for swapctl."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import bootstrap, downloader
from .config import settings
from .routers import api, auth, benchmark, stt, tts, views, public
from .routers.api_keys_router import router as api_keys_router

APP_DIR = Path(__file__).resolve().parent
log = logging.getLogger("swapctl")

PREFIX = settings.root_path
_PUBLIC_EXACT = {
    f"{PREFIX}/public/models",
    f"{PREFIX}/login",
    f"{PREFIX}/logout",
    f"{PREFIX}/healthz",
    f"{PREFIX}/favicon.ico",
    f"{PREFIX}/api/v1/audio/voices",
    f"{PREFIX}/api/v1/audio/speech",
    f"{PREFIX}/api/v1/audio/transcriptions",
}
_PUBLIC_PREFIXES = (f"{PREFIX}/assets/",)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.password:
        log.warning("SWAPCTL_PASSWORD is not set — all requests will be rejected")
    if PREFIX:
        log.info("swapctl mounted under %s", PREFIX)
    try:
        added = bootstrap.sync_from_config()
        if added:
            log.info("registry bootstrapped %d model(s) from llama-swap config", added)
    except Exception as exc:
        log.warning("registry bootstrap skipped: %s", exc)
    try:
        n = downloader.load_persisted()
        if n:
            log.info("loaded %d persisted job record(s); active jobs marked interrupted", n)
    except Exception as exc:
        log.warning("job persistence load skipped: %s", exc)
    try:
        resumed = 0
        for j in list(downloader.list_jobs()):
            if j.get("status") == "interrupted" and j.get("error", "").startswith("swapctl restarted"):
                try:
                    downloader.resume_job(j["job_id"])
                    resumed += 1
                except Exception as exc:
                    log.warning("auto-resume failed for %s: %s", j.get("job_id"), exc)
        if resumed:
            log.info("auto-resumed %d job(s) interrupted by restart", resumed)
    except Exception as exc:
        log.warning("auto-resume skipped: %s", exc)
    try:
        from . import auto_update
        auto_update.start_scheduler()
    except Exception as exc:
        log.warning("auto-update scheduler failed to start: %s", exc)
    try:
        from .backends_registry import BackendsRegistry
        BackendsRegistry().seed_defaults_if_empty()
        try:
            BackendsRegistry().scan_binaries()
        except Exception:
            log.warning("backend binary auto-scan failed", exc_info=True)
    except Exception as exc:
        log.warning("backends registry seed skipped: %s", exc)
    try:
        from . import power, registry
        cfg = registry.get_socket_tdp_config()
        if cfg.get("apply_on_boot"):
            result = power.set_socket_tdp(
                cfg["stapm_mw"], cfg["fast_mw"], cfg["slow_mw"], cfg["apu_mw"],
                stapm_time_s=cfg.get("stapm_time_s"),
            )
            if result.get("ok"):
                log.info("socket TDP restored on boot: %s", result.get("actual"))
            else:
                log.warning("socket TDP restore failed: %s", result.get("error") or result.get("errors"))
    except Exception as exc:
        log.warning("socket TDP boot-apply skipped: %s", exc)
    try:
        from . import background
        background.start_worker()
    except Exception as exc:
        log.warning("snapshot worker failed to start: %s", exc)
    yield


_UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _origin_host(origin: str) -> str:
    """Host[:port] z hlavičky Origin. Prázdné, když je hodnota nepoužitelná."""
    if not origin or origin == "null":
        return ""
    return origin.split("://", 1)[-1].split("/", 1)[0].strip().lower()


def _origin_allowed(origin: str, host: str, forwarded_host: str, extra: str) -> bool:
    """Je zapisující request ze stejného webu?

    Cookie je SameSite=None, aby šla stránka vložit do rámu, takže SameSite
    už zapisující requesty nechrání. Tuhle roli přebírá tahle kontrola:
    prohlížeč u POST/PUT/PATCH/DELETE posílá Origin vždy, takže cizí web se
    tady zastaví. Chybějící Origin znamená klient, který není prohlížeč
    (curl, skript) — ten cizí cookie nenese, takže projde.
    """
    src = _origin_host(origin)
    if not src:
        return True
    allowed = {h.strip().lower() for h in (host, forwarded_host) if h and h.strip()}
    for item in (extra or "").replace(",", " ").split():
        allowed.add(_origin_host(item) or item.strip().lower())
    return src in allowed


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def create_app() -> FastAPI:
    app = FastAPI(title="swapctl", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def no_cache_html(request: Request, call_next):
        resp = await call_next(request)
        ct = resp.headers.get("content-type", "")
        if "text/html" in ct:
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
        return resp

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        path = request.url.path
        if _is_public(path):
            return await call_next(request)
        if not settings.password:
            return JSONResponse(
                {"detail": "SWAPCTL_PASSWORD is not set. Edit /etc/swapctl.env."},
                status_code=503,
            )
        if not request.session.get("authed"):
            if "text/html" in request.headers.get("accept", "").lower():
                return RedirectResponse(f"{PREFIX}/login", status_code=303)
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):
        if request.method in _UNSAFE and not _origin_allowed(
            request.headers.get("origin", ""),
            request.headers.get("host", ""),
            request.headers.get("x-forwarded-host", ""),
            settings.allowed_origins,
        ):
            return JSONResponse(
                {"detail": "cross-site request rejected"}, status_code=403
            )
        return await call_next(request)

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="swapctl_session",
        max_age=settings.session_hours * 3600,
        https_only=settings.cookie_secure,
        # None kvůli vkládání do rámu (prezentace). Zapisující requesty
        # chrání csrf_guard výše, ne SameSite.
        same_site="none",
    )

    app.mount(
        f"{PREFIX}/assets",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )
    app.include_router(auth.router, prefix=PREFIX)
    app.include_router(views.router, prefix=PREFIX)
    app.include_router(public.router, prefix=PREFIX)
    app.include_router(api.router, prefix=PREFIX)
    app.include_router(benchmark.router, prefix=PREFIX)
    app.include_router(tts.router, prefix=f"{PREFIX}/api")
    app.include_router(stt.router, prefix=f"{PREFIX}/api")
    app.include_router(api_keys_router, prefix=PREFIX)

    @app.get(f"{PREFIX}/healthz")
    def healthz():
        return {"ok": True}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
