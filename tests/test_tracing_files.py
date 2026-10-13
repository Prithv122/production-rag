"""The local Langfuse stack's files are checked for the properties that matter, as text.

Nothing here starts a container. These pin the three promises the compose file makes in
its header: every image is pinned, every published port is loopback-only, and the
committed environment template carries no real secret.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = (ROOT / "tracing.compose.yml").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / "tracing.env.example").read_text(encoding="utf-8")


def test_every_image_is_pinned_to_an_exact_tag_or_digest():
    images = re.findall(r"^\s*image:\s*(\S+)", COMPOSE, flags=re.MULTILINE)
    assert len(images) == 6
    for image in images:
        assert "@sha256:" in image or re.search(r":\d+(\.\d+)+(-[\w.]+)?$", image), image
        assert not image.endswith(":latest")


def test_every_published_port_is_bound_to_loopback():
    ports = re.findall(r'^\s*-\s*"([^"]*:\d+:\d+)"', COMPOSE, flags=re.MULTILINE)
    assert ports
    assert all(port.startswith("127.0.0.1:") for port in ports), ports


def test_required_secrets_have_no_default_in_the_compose_file():
    for name in ("POSTGRES_PASSWORD", "CLICKHOUSE_PASSWORD", "REDIS_AUTH", "ENCRYPTION_KEY"):
        assert f"${{{name}:?" in COMPOSE, name
        assert f"${{{name}:-" not in COMPOSE, name


def test_the_env_template_holds_placeholders_not_secrets():
    values = {}
    for line in ENV_EXAMPLE.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    secrets = [
        k for k in values if any(w in k for w in ("PASSWORD", "SECRET", "KEY", "SALT", "AUTH"))
    ]
    assert secrets
    for key in secrets:
        assert "REPLACE_ME" in values[key], key


def test_the_real_env_file_is_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "tracing.env" in [line.strip() for line in ignored]
