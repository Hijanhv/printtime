"""API keys from .env, without ever printing them.

Keys live in .env (gitignored). Code asks for a key by its environment
variable name; a missing key raises an error that names the variable but
never echoes a value.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


class MissingKeyError(RuntimeError):
    """Raised when a required API key is not set."""


def require_key(env_var: str) -> str:
    load_dotenv()
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise MissingKeyError(f"{env_var} is not set. Add it to .env (see .env.example).")
    return value


def has_key(env_var: str) -> bool:
    load_dotenv()
    return bool(os.environ.get(env_var, "").strip())


def redact(value: str) -> str:
    """For logs: show that a key exists without revealing it."""
    return "<set>" if value else "<missing>"
