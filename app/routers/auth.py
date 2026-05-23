"""Password login / logout for swapctl."""
from __future__ import annotations

import asyncio
import hmac
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import settings

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["root_path"] = settings.root_path
try:
    from .. import github_watch
    templates.env.globals["gh_watch"] = github_watch.load_state
except Exception:
    pass

router = APIRouter(tags=["auth"])

PREFIX = settings.root_path


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> HTMLResponse:
    if request.session.get("authed"):
        return RedirectResponse(f"{PREFIX}/", status_code=303)
    if not settings.password:
        raise HTTPException(
            status_code=503,
            detail="SWAPCTL_PASSWORD is not set. Edit /etc/swapctl.env and restart.",
        )
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)) -> HTMLResponse:
    if not settings.password:
        raise HTTPException(status_code=503, detail="SWAPCTL_PASSWORD is not set.")
    if not hmac.compare_digest(password, settings.password):
        await asyncio.sleep(1.0)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid password."},
            status_code=401,
        )
    request.session["authed"] = True
    return RedirectResponse(f"{PREFIX}/", status_code=303)


@router.post("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(f"{PREFIX}/login", status_code=303)
