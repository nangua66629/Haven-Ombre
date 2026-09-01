import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from oauth_dynamic import DynamicOAuthProvider


class FakeRequest:
    def __init__(
        self,
        *,
        method="GET",
        query=None,
        json_body=None,
        form_body=b"",
        headers=None,
        host="198.51.100.10",
    ):
        self.method = method
        self.query_params = query or {}
        self._json_body = json_body
        self._form_body = form_body
        self.headers = headers or {}
        self.client = SimpleNamespace(host=host)
        self.url = SimpleNamespace(scheme="https", netloc="ombre.example")

    async def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body

    async def body(self):
        return self._form_body


def payload(response):
    return json.loads(response.body.decode("utf-8"))


def form_request(data):
    from urllib.parse import urlencode

    return FakeRequest(
        method="POST",
        form_body=urlencode(data).encode("utf-8"),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )


@pytest.mark.asyncio
async def test_dcr_pkce_authorization_and_refresh_survive_restart(tmp_path):
    provider = DynamicOAuthProvider(
        state_dir=str(tmp_path),
        public_base_url="https://ombre.example",
    )
    callback = "https://chatgpt.com/connector/oauth/test-callback"
    registered = await provider.register(
        FakeRequest(
            method="POST",
            json_body={"redirect_uris": [callback], "client_name": "ChatGPT"},
            headers={"content-type": "application/json"},
        )
    )
    assert registered.status_code == 201
    client_id = payload(registered)["client_id"]

    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    authorize_params = {
        "client_id": client_id,
        "redirect_uri": callback,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "mcp",
        "resource": "https://ombre.example/mcp",
        "state": "state-1",
    }
    page = await provider.authorize(
        FakeRequest(query=authorize_params),
        verify_password=lambda value: value == "correct horse",
        setup_needed=lambda: False,
    )
    assert page.status_code == 200
    assert b"Dashboard" in page.body

    approved = await provider.authorize(
        form_request({**authorize_params, "password": "correct horse"}),
        verify_password=lambda value: value == "correct horse",
        setup_needed=lambda: False,
    )
    assert approved.status_code == 302
    location = approved.headers["location"]
    assert location.startswith(callback)
    code = location.split("code=", 1)[1].split("&", 1)[0]

    token_response = await provider.token(
        form_request(
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "redirect_uri": callback,
                "code_verifier": verifier,
                "resource": "https://ombre.example/mcp",
            }
        )
    )
    assert token_response.status_code == 200
    tokens = payload(token_response)
    assert provider.valid_access_token(tokens["access_token"])

    restarted = DynamicOAuthProvider(
        state_dir=str(tmp_path),
        public_base_url="https://ombre.example",
    )
    assert restarted.valid_client_id(client_id)
    assert restarted.valid_access_token(tokens["access_token"])
    refreshed = await restarted.token(
        form_request(
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": tokens["refresh_token"],
                "resource": "https://ombre.example/mcp",
            }
        )
    )
    assert refreshed.status_code == 200
    refreshed_tokens = payload(refreshed)
    assert refreshed_tokens["access_token"] != tokens["access_token"]
    assert not restarted.valid_refresh_token(tokens["refresh_token"])


@pytest.mark.asyncio
async def test_authorization_rejects_wrong_password_and_bad_redirect(tmp_path):
    provider = DynamicOAuthProvider(
        state_dir=str(tmp_path), public_base_url="https://ombre.example"
    )
    callback = "https://chatgpt.com/connector/oauth/callback"
    response = await provider.register(
        FakeRequest(
            method="POST",
            json_body={"redirect_uris": [callback]},
            headers={"content-type": "application/json"},
        )
    )
    client_id = payload(response)["client_id"]
    base = {
        "client_id": client_id,
        "redirect_uri": callback,
        "response_type": "code",
        "code_challenge": "c" * 43,
        "code_challenge_method": "S256",
        "scope": "mcp",
        "resource": "https://ombre.example/mcp",
    }
    wrong_password = await provider.authorize(
        form_request({**base, "password": "wrong"}),
        verify_password=lambda value: False,
        setup_needed=lambda: False,
    )
    assert wrong_password.status_code == 401

    bad_redirect = await provider.authorize(
        FakeRequest(query={**base, "redirect_uri": "https://evil.example/callback"}),
        verify_password=lambda value: True,
        setup_needed=lambda: False,
    )
    assert bad_redirect.status_code == 400
    assert payload(bad_redirect)["error"] == "invalid_request"


def test_metadata_advertises_dcr_and_pkce(tmp_path):
    provider = DynamicOAuthProvider(
        state_dir=str(tmp_path), public_base_url="https://ombre.example"
    )
    metadata = provider.server_metadata(FakeRequest())
    assert metadata["registration_endpoint"] == "https://ombre.example/oauth/register"
    assert metadata["code_challenge_methods_supported"] == ["S256"]
    assert metadata["token_endpoint_auth_methods_supported"] == ["none"]
