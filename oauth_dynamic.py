"""Small OAuth 2.1/DCR provider for the Haven MCP endpoint.

This module keeps OAuth state separate from memory buckets while persisting
clients and grants in the mounted state directory.  It implements the subset
needed by ChatGPT/Codex MCP connections: discovery, dynamic client
registration, authorization-code + PKCE, refresh-token rotation, and Bearer
validation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlsplit

from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse


class DynamicOAuthProvider:
    CODE_TTL = 300
    ACCESS_TOKEN_TTL = 30 * 24 * 60 * 60
    REFRESH_TOKEN_TTL = 365 * 24 * 60 * 60
    CLIENT_TTL = 365 * 24 * 60 * 60
    MAX_CLIENTS = 256
    MAX_REDIRECT_URIS = 10
    MAX_REDIRECT_URI_CHARS = 2048
    REGISTRATION_WINDOW = 60
    REGISTRATION_PER_SOURCE = 10
    REGISTRATION_GLOBAL = 120

    def __init__(
        self,
        *,
        state_dir: str,
        public_base_url: str = "",
        fixed_client_id: str = "",
        fixed_client_secret: str = "",
        fixed_access_token: str = "",
        fixed_refresh_token: str = "",
        dynamic_enabled: bool = True,
    ) -> None:
        self.state_path = Path(state_dir) / ".mcp_oauth_state.json"
        self.public_base_url = str(public_base_url or "").strip().rstrip("/")
        self.fixed_client_id = str(fixed_client_id or "").strip()
        self.fixed_client_secret = str(fixed_client_secret or "").strip()
        self.fixed_access_token = str(fixed_access_token or "").strip()
        self.fixed_refresh_token = str(fixed_refresh_token or "").strip()
        self.dynamic_enabled = bool(dynamic_enabled)
        self._lock = threading.RLock()
        self._clients: dict[str, dict[str, Any]] = {}
        self._codes: dict[str, dict[str, Any]] = {}
        self._access_tokens: dict[str, dict[str, Any]] = {}
        self._refresh_tokens: dict[str, dict[str, Any]] = {}
        self._registration_by_source: dict[str, deque[float]] = defaultdict(deque)
        self._registration_global: deque[float] = deque()
        self._load()

    @property
    def enabled(self) -> bool:
        return self.dynamic_enabled or bool(
            self.fixed_client_id and self.fixed_access_token
        )

    @property
    def token_auth_methods(self) -> list[str]:
        methods = ["none"]
        if self.fixed_client_secret:
            methods.extend(["client_secret_post", "client_secret_basic"])
        return methods

    def external_base(self, request=None) -> str:
        if self.public_base_url:
            return self.public_base_url
        if request is None:
            return ""
        headers = request.headers
        proto = str(headers.get("x-forwarded-proto") or request.url.scheme or "https")
        proto = proto.split(",", 1)[0].strip().lower()
        if proto not in {"http", "https"}:
            proto = "https"
        host = str(headers.get("x-forwarded-host") or headers.get("host") or request.url.netloc)
        host = host.split(",", 1)[0].strip()
        return f"{proto}://{host}".rstrip("/")

    def server_metadata(self, request) -> dict[str, Any]:
        base = self.external_base(request)
        return {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "registration_endpoint": f"{base}/oauth/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": self.token_auth_methods,
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["mcp"],
        }

    def resource_metadata(self, request) -> dict[str, Any]:
        base = self.external_base(request)
        return {
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "scopes_supported": ["mcp"],
            "bearer_methods_supported": ["header"],
        }

    def challenge(self, request=None) -> str:
        base = self.external_base(request)
        metadata = f"{base}/.well-known/oauth-protected-resource" if base else "/.well-known/oauth-protected-resource"
        return f'Bearer resource_metadata="{metadata}", scope="mcp"'

    def challenge_for_scope(self, scope: dict[str, Any]) -> str:
        if self.public_base_url:
            base = self.public_base_url
        else:
            headers = {
                key.decode("latin1").lower(): value.decode("latin1")
                for key, value in scope.get("headers", [])
            }
            proto = str(headers.get("x-forwarded-proto") or scope.get("scheme") or "https")
            proto = proto.split(",", 1)[0].strip().lower()
            if proto not in {"http", "https"}:
                proto = "https"
            host = str(headers.get("x-forwarded-host") or headers.get("host") or "")
            host = host.split(",", 1)[0].strip()
            base = f"{proto}://{host}" if host else ""
        metadata = (
            f"{base}/.well-known/oauth-protected-resource"
            if base
            else "/.well-known/oauth-protected-resource"
        )
        return f'Bearer resource_metadata="{metadata}", scope="mcp"'

    async def register(self, request):
        if not self.dynamic_enabled:
            return self._error("registration_not_supported", 404)
        retry_after = self._reserve_registration(self._source_key(request))
        if retry_after:
            return JSONResponse(
                {"error": "slow_down", "error_description": "registration rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
            )
        try:
            body = await request.json()
        except Exception:
            return self._error("invalid_client_metadata", 400)
        registration, error = self._normalize_registration(body)
        if error:
            return self._error("invalid_client_metadata", 400, error)

        with self._lock:
            self._cleanup_locked()
            if len(self._clients) >= self.MAX_CLIENTS:
                return self._error("temporarily_unavailable", 429, "client registry is full")
            client_id = f"ombre_{secrets.token_urlsafe(24)}"
            now = time.time()
            self._clients[client_id] = {
                **registration,
                "created_at": now,
                "expires_at": now + self.CLIENT_TTL,
            }
            self._save_locked()

        return JSONResponse(
            {
                "client_id": client_id,
                "client_id_issued_at": int(now),
                "client_name": registration["client_name"],
                "redirect_uris": registration["redirect_uris"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
            headers={"Cache-Control": "no-store"},
        )

    async def authorize(
        self,
        request,
        *,
        verify_password: Callable[[str], bool],
        setup_needed: Callable[[], bool],
    ):
        params = await self._request_params(request)
        validated, error = self._validate_authorize(params, request)
        if error:
            return self._error(error[0], error[1], error[2])

        if request.method.upper() == "GET":
            return self._authorization_page(validated, setup_needed=setup_needed())

        password = str(params.get("password") or "")
        if setup_needed():
            return self._error("access_denied", 403, "dashboard password is not configured")
        if not verify_password(password):
            return self._authorization_page(
                validated,
                setup_needed=False,
                error_message="密码不正确，请重试。",
                status_code=401,
            )

        code = secrets.token_urlsafe(32)
        with self._lock:
            self._cleanup_locked()
            self._codes[code] = {
                "client_id": validated["client_id"],
                "redirect_uri": validated["redirect_uri"],
                "code_challenge": validated["code_challenge"],
                "resource": validated["resource"],
                "scope": validated["scope"],
                "expires_at": time.time() + self.CODE_TTL,
            }

        query = {"code": code}
        if validated.get("state"):
            query["state"] = validated["state"]
        separator = "&" if "?" in validated["redirect_uri"] else "?"
        return RedirectResponse(
            f"{validated['redirect_uri']}{separator}{urlencode(query)}",
            status_code=302,
            headers={"Cache-Control": "no-store"},
        )

    async def token(self, request):
        form = await self._request_params(request)
        basic_id, basic_secret = self._basic_credentials(request.headers)
        client_id = basic_id or str(form.get("client_id") or "")
        client_secret = basic_secret or str(form.get("client_secret") or "")
        if not self.valid_client_id(client_id):
            return self._error("invalid_client", 401)
        if not self.valid_client_secret(client_id, client_secret):
            return self._error("invalid_client", 401)

        grant_type = str(form.get("grant_type") or "")
        if grant_type == "authorization_code":
            return self._exchange_code(client_id, form)
        if grant_type == "refresh_token":
            return self._exchange_refresh(client_id, form)
        return self._error("unsupported_grant_type", 400)

    def valid_client_id(self, client_id: str | None) -> bool:
        value = str(client_id or "")
        if self.fixed_client_id and hmac.compare_digest(value, self.fixed_client_id):
            return True
        with self._lock:
            self._cleanup_locked()
            return value in self._clients

    def valid_client_secret(self, client_id: str | None, secret: str | None) -> bool:
        value = str(client_id or "")
        if self.fixed_client_id and hmac.compare_digest(value, self.fixed_client_id):
            if not self.fixed_client_secret:
                return True
            return bool(secret) and hmac.compare_digest(str(secret), self.fixed_client_secret)
        return not secret

    def valid_redirect_uri(self, client_id: str | None, redirect_uri: str | None) -> bool:
        value = str(redirect_uri or "")
        if self.fixed_client_id and hmac.compare_digest(str(client_id or ""), self.fixed_client_id):
            return self._safe_redirect(value)
        with self._lock:
            client = self._clients.get(str(client_id or ""))
            return bool(client and value in client.get("redirect_uris", []))

    def valid_access_token(self, token: str | None) -> bool:
        value = str(token or "")
        if self.fixed_access_token and hmac.compare_digest(value, self.fixed_access_token):
            return True
        with self._lock:
            self._cleanup_locked()
            return value in self._access_tokens

    def valid_refresh_token(self, token: str | None) -> bool:
        value = str(token or "")
        if self.fixed_refresh_token and hmac.compare_digest(value, self.fixed_refresh_token):
            return True
        with self._lock:
            self._cleanup_locked()
            return value in self._refresh_tokens

    def _exchange_code(self, client_id: str, form: dict[str, str]):
        code = str(form.get("code") or "")
        redirect_uri = str(form.get("redirect_uri") or "")
        verifier = str(form.get("code_verifier") or "")
        with self._lock:
            self._cleanup_locked()
            grant = self._codes.get(code)
            if not grant:
                return self._error("invalid_grant", 400)
            if grant["client_id"] != client_id or grant["redirect_uri"] != redirect_uri:
                return self._error("invalid_grant", 400)
            requested_resource = str(form.get("resource") or grant["resource"])
            if requested_resource != grant["resource"]:
                return self._error("invalid_target", 400)
            if not self._verify_pkce(verifier, grant["code_challenge"]):
                return self._error("invalid_grant", 400, "PKCE verification failed")
            self._codes.pop(code, None)
            return self._issue_tokens_locked(client_id, grant["resource"], grant["scope"])

    def _exchange_refresh(self, client_id: str, form: dict[str, str]):
        refresh_token = str(form.get("refresh_token") or "")
        with self._lock:
            self._cleanup_locked()
            grant = self._refresh_tokens.get(refresh_token)
            if not grant or grant["client_id"] != client_id:
                return self._error("invalid_grant", 400)
            requested_resource = str(form.get("resource") or grant["resource"])
            if requested_resource != grant["resource"]:
                return self._error("invalid_target", 400)
            self._refresh_tokens.pop(refresh_token, None)
            return self._issue_tokens_locked(client_id, grant["resource"], grant["scope"])

    def _issue_tokens_locked(self, client_id: str, resource: str, scope: str):
        now = time.time()
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(40)
        self._access_tokens[access_token] = {
            "client_id": client_id,
            "resource": resource,
            "scope": scope,
            "expires_at": now + self.ACCESS_TOKEN_TTL,
        }
        self._refresh_tokens[refresh_token] = {
            "client_id": client_id,
            "resource": resource,
            "scope": scope,
            "expires_at": now + self.REFRESH_TOKEN_TTL,
        }
        self._save_locked()
        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": self.ACCESS_TOKEN_TTL,
                "refresh_token": refresh_token,
                "scope": scope,
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    def _validate_authorize(self, params: dict[str, str], request):
        client_id = str(params.get("client_id") or "")
        redirect_uri = str(params.get("redirect_uri") or "")
        response_type = str(params.get("response_type") or "")
        challenge = str(params.get("code_challenge") or "")
        challenge_method = str(params.get("code_challenge_method") or "")
        scope = str(params.get("scope") or "mcp")
        base = self.external_base(request)
        resource = str(params.get("resource") or f"{base}/mcp").rstrip("/")
        canonical_resource = f"{base}/mcp".rstrip("/")
        if response_type != "code":
            return None, ("unsupported_response_type", 400, "")
        if not self.valid_client_id(client_id):
            return None, ("invalid_client", 401, "")
        if not self.valid_redirect_uri(client_id, redirect_uri):
            return None, ("invalid_request", 400, "redirect_uri is not registered")
        if challenge_method != "S256" or not 43 <= len(challenge) <= 128:
            return None, ("invalid_request", 400, "S256 PKCE is required")
        if set(scope.split()) != {"mcp"}:
            return None, ("invalid_scope", 400, "")
        if resource != canonical_resource:
            return None, ("invalid_target", 400, "")
        return {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": response_type,
            "code_challenge": challenge,
            "code_challenge_method": challenge_method,
            "scope": "mcp",
            "resource": canonical_resource,
            "state": str(params.get("state") or ""),
        }, None

    def _authorization_page(
        self,
        params: dict[str, str],
        *,
        setup_needed: bool,
        error_message: str = "",
        status_code: int = 200,
    ):
        client = self._clients.get(params["client_id"], {})
        client_name = str(client.get("client_name") or "MCP 客户端")
        hidden = "".join(
            f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(str(value), quote=True)}">'
            for key, value in params.items()
            if key != "password"
        )
        warning = (
            '<p class="error">Dashboard 密码尚未配置，请先完成 Dashboard 初始化。</p>'
            if setup_needed
            else (f'<p class="error">{html.escape(error_message)}</p>' if error_message else "")
        )
        disabled = " disabled" if setup_needed else ""
        page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>连接 Ombre Brain</title><style>
body{{font-family:system-ui,sans-serif;background:#f5f1eb;color:#2d2926;margin:0;padding:32px}}
.card{{max-width:460px;margin:8vh auto;background:white;border-radius:18px;padding:28px;box-shadow:0 12px 36px #0002}}
input{{box-sizing:border-box;width:100%;padding:12px;margin:10px 0;border:1px solid #bbb;border-radius:10px}}
button{{width:100%;padding:12px;border:0;border-radius:10px;background:#9a6b48;color:white;font-weight:700}}
.muted{{color:#746b65;word-break:break-all}}.error{{color:#a32626}}
</style></head><body><main class="card"><h1>连接 Ombre Brain</h1>
<p><strong>{html.escape(client_name)}</strong> 请求访问你的私人记忆工具。</p>
<p class="muted">回调地址：{html.escape(params['redirect_uri'])}</p>{warning}
<form method="post" action="/oauth/authorize">{hidden}
<label>Dashboard 密码</label><input type="password" name="password" autocomplete="current-password" required{disabled}>
<button type="submit"{disabled}>授权连接</button></form></main></body></html>"""
        return HTMLResponse(page, status_code=status_code, headers={"Cache-Control": "no-store"})

    def _normalize_registration(self, body: Any):
        if not isinstance(body, dict):
            return None, "body must be a JSON object"
        redirect_uris = body.get("redirect_uris")
        if (
            not isinstance(redirect_uris, list)
            or not 1 <= len(redirect_uris) <= self.MAX_REDIRECT_URIS
            or any(not self._safe_redirect(uri) for uri in redirect_uris)
        ):
            return None, "redirect_uris must contain safe absolute URLs"
        client_name = str(body.get("client_name") or "MCP Client").strip()[:200]
        return {
            "redirect_uris": list(dict.fromkeys(str(uri) for uri in redirect_uris)),
            "client_name": client_name or "MCP Client",
        }, ""

    @classmethod
    def _safe_redirect(cls, value: Any) -> bool:
        if not isinstance(value, str) or not 1 <= len(value) <= cls.MAX_REDIRECT_URI_CHARS:
            return False
        try:
            parsed = urlsplit(value)
        except ValueError:
            return False
        if parsed.fragment or parsed.username or parsed.password:
            return False
        if parsed.scheme == "https":
            return bool(parsed.hostname)
        return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _verify_pkce(verifier: str, challenge: str) -> bool:
        if not 43 <= len(verifier) <= 128:
            return False
        try:
            encoded = verifier.encode("ascii", "strict")
        except UnicodeEncodeError:
            return False
        digest = hashlib.sha256(encoded).digest()
        calculated = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return hmac.compare_digest(calculated, challenge)

    @staticmethod
    async def _request_params(request) -> dict[str, str]:
        if request.method.upper() == "GET":
            return {str(k): str(v) for k, v in request.query_params.items()}
        content_type = str(request.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    return {str(k): str(v) for k, v in body.items()}
            except Exception:
                return {}
        body = await request.body()
        parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        return {str(k): str(values[-1] if values else "") for k, values in parsed.items()}

    @staticmethod
    def _basic_credentials(headers) -> tuple[str, str]:
        auth = str(headers.get("authorization") or "")
        if not auth.lower().startswith("basic "):
            return "", ""
        try:
            decoded = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
            return tuple(decoded.split(":", 1))  # type: ignore[return-value]
        except Exception:
            return "", ""

    def _reserve_registration(self, source: str) -> int:
        now = time.time()
        cutoff = now - self.REGISTRATION_WINDOW
        with self._lock:
            while self._registration_global and self._registration_global[0] <= cutoff:
                self._registration_global.popleft()
            attempts = self._registration_by_source[source]
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if len(attempts) >= self.REGISTRATION_PER_SOURCE or len(self._registration_global) >= self.REGISTRATION_GLOBAL:
                oldest = attempts[0] if attempts else self._registration_global[0]
                return max(1, int(oldest + self.REGISTRATION_WINDOW - now) + 1)
            attempts.append(now)
            self._registration_global.append(now)
            return 0

    @staticmethod
    def _source_key(request) -> str:
        client = getattr(request, "client", None)
        return str(getattr(client, "host", "unknown") or "unknown")

    def _cleanup_locked(self) -> None:
        now = time.time()
        for mapping in (self._clients, self._codes, self._access_tokens, self._refresh_tokens):
            for key, value in list(mapping.items()):
                if not isinstance(value, dict) or float(value.get("expires_at") or 0) <= now:
                    mapping.pop(key, None)

    def _load(self) -> None:
        try:
            if not self.state_path.exists():
                return
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            self._clients = dict(raw.get("clients") or {})
            self._access_tokens = dict(raw.get("access_tokens") or {})
            self._refresh_tokens = dict(raw.get("refresh_tokens") or {})
            self._cleanup_locked()
        except Exception:
            self._clients = {}
            self._access_tokens = {}
            self._refresh_tokens = {}

    def _save_locked(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clients": self._clients,
            "access_tokens": self._access_tokens,
            "refresh_tokens": self._refresh_tokens,
        }
        temp_path = self.state_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, self.state_path)

    @staticmethod
    def _error(code: str, status: int, description: str = ""):
        payload = {"error": code}
        if description:
            payload["error_description"] = description
        return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})
