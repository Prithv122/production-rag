"""Secret lookup: the environment first, then the OS credential store.

On this machine the credential store is Windows Credential Manager, filled once with
``keyring set deepbrief ANTHROPIC_API_KEY``. One shared service name serves every project;
set ``SECRETS_SERVICE`` to use a different one.

The environment wins, so CI, Cloud Run and Hugging Face Spaces work unchanged. Keyring is an
optional extra: if it is not installed, or the machine has no credential backend, a secret
simply reads as "not set". Tests set ``SECRETS_NO_KEYRING=1`` so a key stored on the
developer's machine can never turn a replay or "no key" test into a live call.
"""

from __future__ import annotations

import os

DEFAULT_SERVICE = "deepbrief"


def get_secret(name: str, service: str | None = None) -> str | None:
    """Return the secret called ``name``, or None when it is not set anywhere."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if os.environ.get("SECRETS_NO_KEYRING") == "1":
        return None
    try:
        import keyring

        stored = keyring.get_password(
            service or os.environ.get("SECRETS_SERVICE", DEFAULT_SERVICE), name
        )
    except Exception:  # keyring not installed, or no backend (CI, Linux container)
        return None
    return (stored or "").strip() or None
