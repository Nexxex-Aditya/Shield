"""
Shield — Authentication & API Key Management
JWT-based auth with bcrypt password hashing and cryptographic API keys.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr

logger = logging.getLogger("shield.auth")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JWT_SECRET = os.getenv("SHIELD_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("SHIELD_JWT_EXPIRY_HOURS", "72"))

auth_router = APIRouter(prefix="/auth", tags=["auth"])

# Module-level reference to DB — set by app.py at startup
_db = None


def set_auth_db(db):
    """Called by app.py to inject the database store."""
    global _db
    _db = db


# ---------------------------------------------------------------------------
# Password hashing (using hashlib — no extra C deps needed)
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return check.hex() == h
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


async def get_current_user(request: Request) -> dict:
    """FastAPI dependency — extracts and validates JWT from Authorization header."""
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = _decode_token(token)
        return {"id": payload["sub"], "email": payload["email"]}

    # Also check API key
    if auth_header.startswith("sk-shield-"):
        if not _db:
            raise HTTPException(500, "Database not initialized")
        user = await _db.verify_api_key(auth_header)
        if not user:
            raise HTTPException(401, "Invalid API key")
        return user

    raise HTTPException(401, "Missing or invalid authorization")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ApiKeyCreateRequest(BaseModel):
    name: str = "Default"


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@auth_router.post("/signup")
async def signup(req: SignupRequest):
    """Create a new user account."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    # Check if user exists
    existing = await _db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(409, "Email already registered")

    user_id = str(uuid.uuid4())
    password_hash = _hash_password(req.password)

    await _db.create_user({
        "id": user_id,
        "email": req.email,
        "name": req.name or req.email.split("@")[0],
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    token = _create_token(user_id, req.email)
    logger.info("User signed up: %s", req.email)

    return {
        "token": token,
        "user": {"id": user_id, "email": req.email, "name": req.name},
    }


@auth_router.post("/login")
async def login(req: LoginRequest):
    """Authenticate and return JWT."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    user = await _db.get_user_by_email(req.email)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    token = _create_token(user["id"], user["email"])
    logger.info("User logged in: %s", req.email)

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name", ""),
        },
    }


@auth_router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user profile."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    full_user = await _db.get_user_by_email(user["email"])
    if not full_user:
        raise HTTPException(404, "User not found")

    return {
        "id": full_user["id"],
        "email": full_user["email"],
        "name": full_user.get("name", ""),
        "created_at": full_user.get("created_at", ""),
    }


@auth_router.post("/keys")
async def create_api_key(
    req: ApiKeyCreateRequest,
    user: dict = Depends(get_current_user),
):
    """Generate a new API key for the authenticated user."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    # Generate key: sk-shield-{random}
    raw_key = f"sk-shield-{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    prefix = raw_key[:18] + "..."
    key_id = str(uuid.uuid4())

    await _db.save_api_key({
        "id": key_id,
        "user_id": user["id"],
        "key_hash": key_hash,
        "name": req.name,
        "prefix": prefix,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info("API key created for user %s", user["email"])

    # Return the raw key ONCE — it's never stored in plaintext
    return {
        "key": raw_key,
        "id": key_id,
        "name": req.name,
        "prefix": prefix,
    }


@auth_router.get("/keys")
async def list_api_keys(user: dict = Depends(get_current_user)):
    """List all API keys for the authenticated user (prefixes only)."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    keys = await _db.get_api_keys(user["id"])
    return {"keys": keys}


@auth_router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: dict = Depends(get_current_user),
):
    """Revoke an API key."""
    if not _db:
        raise HTTPException(500, "Database not initialized")

    result = await _db.revoke_api_key(key_id, user["id"])
    if not result:
        raise HTTPException(404, "API key not found")

    logger.info("API key revoked: %s", key_id)
    return {"ok": True}
