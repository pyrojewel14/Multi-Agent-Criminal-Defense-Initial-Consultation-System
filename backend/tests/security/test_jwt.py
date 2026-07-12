import os
import time
from datetime import datetime, timedelta

import pytest

# Set env vars before importing JWT module so config picks them up
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-at-least-32-characters-long!!")

from app.security.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry,
    hash_password,
    verify_password,
)


class TestHashAndVerifyPassword:
    """Tests for hash_password + verify_password."""

    def test_hash_then_verify_succeeds(self):
        password = "MySecret123!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_is_different_from_plain(self):
        password = "plain_text"
        hashed = hash_password(password)
        assert hashed != password

    def test_different_hashes_for_same_password(self):
        """bcrypt generates a different salt each time."""
        password = "same_password"
        h1 = hash_password(password)
        h2 = hash_password(password)
        assert h1 != h2
        # But both should verify
        assert verify_password(password, h1) is True
        assert verify_password(password, h2) is True


class TestCreateAndDecodeAccessToken:
    """Tests for create_access_token + decode_token."""

    def test_create_then_decode_returns_correct_payload(self):
        token = create_access_token(user_id="user123", role="lawyer")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["role"] == "lawyer"
        assert payload["type"] == "access"

    def test_token_contains_exp_and_iat(self):
        token = create_access_token(user_id="u1", role="admin")
        payload = decode_token(token)
        assert "exp" in payload
        assert "iat" in payload


class TestCreateRefreshToken:
    """Tests for create_refresh_token."""

    def test_creates_valid_token(self):
        token = create_refresh_token(user_id="user456")
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "user456"
        assert payload["type"] == "refresh"

    def test_refresh_token_has_no_role(self):
        token = create_refresh_token(user_id="user456")
        payload = decode_token(token)
        assert "role" not in payload


class TestGetTokenExpiry:
    """Tests for get_token_expiry."""

    def test_returns_future_datetime(self):
        token = create_access_token(user_id="u1", role="lawyer")
        expiry = get_token_expiry(token)
        assert expiry is not None
        assert expiry > datetime.utcnow()

    def test_returns_none_for_invalid_token(self):
        result = get_token_expiry("not.a.valid.token")
        assert result is None


class TestExpiredToken:
    """Tests for expired token handling."""

    def test_expired_token_returns_none(self):
        import jwt as pyjwt
        from app.security.config import get_jwt_config

        config = get_jwt_config()
        # Create a token that expired 1 hour ago
        payload = {
            "sub": "expired_user",
            "role": "lawyer",
            "type": "access",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2),
        }
        expired_token = pyjwt.encode(
            payload, config.get_secret_key(), algorithm=config.get_algorithm()
        )
        result = decode_token(expired_token)
        assert result is None


class TestInvalidToken:
    """Tests for invalid token handling."""

    def test_invalid_token_returns_none(self):
        result = decode_token("this.is.not.a.jwt")
        assert result is None

    def test_tampered_token_returns_none(self):
        token = create_access_token(user_id="u1", role="lawyer")
        # Tamper with the token by changing a character
        tampered = token[:-5] + "XXXXX"
        result = decode_token(tampered)
        assert result is None
