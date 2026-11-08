from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pwdlib import PasswordHash

AUTH_DB_PATH = Path(os.getenv("SWARM_AUTH_DB_PATH", Path(__file__).with_name("swarm_auth.sqlite3")))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_urlsafe(48)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(hours=8)
PASSWORD_HASHER = PasswordHash.recommended()
DUMMY_PASSWORD_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(24))
BEARER_SCHEME = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    id: int
    email: str
    role: str
    token_version: int

    def public_data(self) -> dict[str, int | str]:
        return {"id": self.id, "email": self.email, "role": self.role}


def _connect() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUTH_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            token_version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS google_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_sub TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            token_version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS google_login_nonces (
            nonce_hash TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            consumed_at REAL
        )
        """
    )
    return connection


def _user_from_row(row: sqlite3.Row | None) -> AuthUser | None:
    if row is None:
        return None
    return AuthUser(
        id=row["id"],
        email=row["email"],
        role=row["role"],
        token_version=row["token_version"],
    )


def create_user(email: str, password: str) -> AuthUser | None:
    normalized_email = email.strip().lower()
    password_hash = PASSWORD_HASHER.hash(password)
    connection = _connect()
    try:
        cursor = connection.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (normalized_email, password_hash),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _user_from_row(row)
    except sqlite3.IntegrityError:
        connection.rollback()
        return None
    finally:
        connection.close()


def get_user_by_email(email: str) -> tuple[AuthUser | None, str | None]:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip().lower(),),
        ).fetchone()
        if row is None:
            return None, None
        return _user_from_row(row), row["password_hash"]
    finally:
        connection.close()


def get_user_by_id(user_id: int) -> AuthUser | None:
    connection = _connect()
    try:
        if user_id < 0:
            row = connection.execute(
                """
                SELECT -id AS id, email, role, token_version
                FROM google_accounts WHERE id = ?
                """,
                (-user_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return _user_from_row(row)
    finally:
        connection.close()


def authenticate_user(email: str, password: str) -> AuthUser | None:
    user, encoded_hash = get_user_by_email(email)
    try:
        password_matches = PASSWORD_HASHER.verify(password, encoded_hash or DUMMY_PASSWORD_HASH)
    except Exception:
        password_matches = False
    return user if user is not None and password_matches else None


def create_google_login_nonce() -> str:
    nonce = secrets.token_urlsafe(32)
    now = time.time()
    connection = _connect()
    try:
        connection.execute("DELETE FROM google_login_nonces WHERE expires_at <= ?", (now,))
        connection.execute(
            "INSERT INTO google_login_nonces (nonce_hash, expires_at) VALUES (?, ?)",
            (hashlib.sha256(nonce.encode("utf-8")).hexdigest(), now + 300),
        )
        connection.commit()
        return nonce
    finally:
        connection.close()


def consume_google_login_nonce(nonce: str) -> bool:
    nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    connection = _connect()
    try:
        cursor = connection.execute(
            """
            UPDATE google_login_nonces SET consumed_at = ?
            WHERE nonce_hash = ? AND consumed_at IS NULL AND expires_at > ?
            """,
            (time.time(), nonce_hash, time.time()),
        )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def authenticate_google_user(email: str, google_sub: str) -> AuthUser | None:
    normalized_email = email.strip().lower()
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        account_row = connection.execute(
            "SELECT * FROM google_accounts WHERE google_sub = ?",
            (google_sub,),
        ).fetchone()
        if account_row is not None:
            connection.commit()
            return AuthUser(
                id=-account_row["id"],
                email=account_row["email"],
                role=account_row["role"],
                token_version=account_row["token_version"],
            )
        cursor = connection.execute(
            "INSERT INTO google_accounts (google_sub, email) VALUES (?, ?)",
            (google_sub, normalized_email),
        )
        connection.commit()
        return AuthUser(
            id=-cursor.lastrowid,
            email=normalized_email,
            role="user",
            token_version=0,
        )
    except sqlite3.IntegrityError:
        connection.rollback()
        row = connection.execute(
            "SELECT * FROM google_accounts WHERE google_sub = ?",
            (google_sub,),
        ).fetchone()
        if row is not None:
            return AuthUser(
                id=-row["id"],
                email=row["email"],
                role=row["role"],
                token_version=row["token_version"],
            )
        return None
    finally:
        connection.close()


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user.id),
            "ver": user.token_version,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def revoke_user_tokens(user_id: int) -> None:
    connection = _connect()
    try:
        if user_id < 0:
            connection.execute(
                "UPDATE google_accounts SET token_version = token_version + 1 WHERE id = ?",
                (-user_id,),
            )
        else:
            connection.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                (user_id,),
            )
        connection.commit()
    finally:
        connection.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(BEARER_SCHEME),
) -> AuthUser:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub", "ver"]},
        )
        user_id = int(payload["sub"])
        token_version = int(payload["ver"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError):
        raise unauthorized

    user = get_user_by_id(user_id)
    if user is None or user.token_version != token_version:
        raise unauthorized
    return user


def validate_email(email: str) -> bool:
    normalized_email = email.strip()
    return len(normalized_email) <= 254 and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_email) is not None
