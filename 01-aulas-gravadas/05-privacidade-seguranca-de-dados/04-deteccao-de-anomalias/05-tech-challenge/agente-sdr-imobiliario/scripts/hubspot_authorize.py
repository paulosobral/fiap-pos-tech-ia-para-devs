"""Autoriza o MCP connector do HubSpot (OAuth 2.1 + PKCE) e grava o refresh token em secrets.local.env.

Uso:  .venv/bin/python scripts/hubspot_authorize.py [--tools]
Sobe um servidor em localhost:6274 (redirect cadastrado no connector), abre o navegador,
troca o code por tokens e lista as tools do MCP (--tools lista sem reautorizar).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

ENV = Path(__file__).resolve().parent.parent / "secrets.local.env"
AUTH_URL = "https://mcp.hubspot.com/oauth/authorize/user"
TOKEN_URL = "https://mcp.hubspot.com/oauth/v3/token"
MCP_URL = "https://mcp.hubspot.com"
REDIRECT = "http://localhost:6274/oauth/callback/debug"


def read_env() -> dict[str, str]:
    out = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def write_env(key: str, value: str) -> None:
    text = ENV.read_text(encoding="utf-8")
    text = re.sub(rf"^{key}=.*$", lambda _m: f"{key}={value}", text, flags=re.M)
    ENV.write_text(text, encoding="utf-8")


def authorize(cfg: dict[str, str]) -> str:
    verifier = secrets.token_urlsafe(64)[:100]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(16)
    box: dict[str, str] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in q and q.get("state", [""])[0] == state:
                box["code"] = q["code"][0]
            else:
                box["error"] = str(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h3>Autorizado. Pode fechar esta aba.</h3>".encode())
            done.set()

        def log_message(self, *a):
            pass

    server = HTTPServer(("localhost", 6274), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": cfg["HUBSPOT_MCP_CLIENT_ID"], "redirect_uri": REDIRECT, "response_type": "code",
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
    })
    print("Abra se o navegador não abrir sozinho:\n" + url)
    webbrowser.open(url)
    if not done.wait(300):
        sys.exit("Tempo esgotado (5 min) sem autorização.")
    server.shutdown()
    if "code" not in box:
        sys.exit(f"Autorização falhou: {box.get('error')}")
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "authorization_code", "client_id": cfg["HUBSPOT_MCP_CLIENT_ID"],
        "client_secret": cfg["HUBSPOT_MCP_CLIENT_SECRET"], "redirect_uri": REDIRECT,
        "code": box["code"], "code_verifier": verifier,
    }, timeout=30)
    if resp.status_code != 200:
        sys.exit(f"Troca de token falhou: {resp.status_code} {resp.text}")
    tokens = resp.json()
    write_env("HUBSPOT_MCP_REFRESH_TOKEN", tokens["refresh_token"])
    print("Refresh token gravado em secrets.local.env")
    return tokens["access_token"]


def refresh(cfg: dict[str, str]) -> str:
    """Refresh token é de uso único: o novo substitui o antigo no arquivo."""
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "refresh_token", "client_id": cfg["HUBSPOT_MCP_CLIENT_ID"],
        "client_secret": cfg["HUBSPOT_MCP_CLIENT_SECRET"], "refresh_token": cfg["HUBSPOT_MCP_REFRESH_TOKEN"],
    }, timeout=30)
    if resp.status_code != 200:
        sys.exit(f"Refresh falhou: {resp.status_code} {resp.text}")
    tokens = resp.json()
    if tokens.get("refresh_token"):
        write_env("HUBSPOT_MCP_REFRESH_TOKEN", tokens["refresh_token"])
    return tokens["access_token"]


async def list_tools(access_token: str) -> None:
    from mcp import ClientSession
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {access_token}"})
    async with streamable_http_client(MCP_URL, http_client=http) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            dump = [{"name": t.name, "description": t.description, "inputSchema": t.input_schema} for t in tools]
            out = Path(__file__).resolve().parent.parent / "hubspot_mcp_tools.json"
            out.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{len(tools)} tools: " + ", ".join(t.name for t in tools))
            print(f"Detalhes em {out.name}")


if __name__ == "__main__":
    cfg = read_env()
    if not cfg.get("HUBSPOT_MCP_CLIENT_ID") or not cfg.get("HUBSPOT_MCP_CLIENT_SECRET"):
        sys.exit("Preencha HUBSPOT_MCP_CLIENT_ID/SECRET em secrets.local.env")
    token = refresh(cfg) if ("--tools" in sys.argv and cfg.get("HUBSPOT_MCP_REFRESH_TOKEN")) else authorize(cfg)
    asyncio.run(list_tools(token))
