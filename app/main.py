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
from .routers import api, auth, benchmark, views

APP_DIR = Path(__file__).resolve().parent
log = logging.getLogger("swapctl")

PREFIX = settings.root_path
_PUBLIC_EXACT = {f"{PREFIX}/login", f"{PREFIX}/logout", f"{PREFIX}/healthz", f"{PREFIX}/favicon.ico"}
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
        from . import auto_update
        auto_update.start_scheduler()
    except Exception as exc:
        log.warning("auto-update scheduler failed to start: %s", exc)
    try:
        from .backends_registry import BackendsRegistry
        BackendsRegistry().seed_defaults_if_empty()
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
    yield


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def create_app() -> FastAPI:
    app = FastAPI(title="swapctl", version="0.1.0", lifespan=lifespan)

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

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="swapctl_session",
        max_age=settings.session_hours * 3600,
        https_only=settings.cookie_secure,
        same_site="lax",
    )

    app.mount(
        f"{PREFIX}/assets",
        StaticFiles(directory=str(APP_DIR / "static")),
        name="static",
    )
    app.include_router(auth.router, prefix=PREFIX)
    app.include_router(views.router, prefix=PREFIX)
    app.include_router(api.router, prefix=PREFIX)
    app.include_router(benchmark.router, prefix=PREFIX)

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
