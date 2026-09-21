"""Every test runs with a made-up key, no key files, a throwaway cache, and outbound connections refused."""

import socket

import pytest

LOCAL = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """A real key may sit in ~/.config/jev, so point the config and cache somewhere empty and block the network."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in ("TYPESAFE_API_KEY", "JEV_API", "JEV_MODEL", "JEV_URL"):
        monkeypatch.delenv(name, raising=False)

    real = socket.socket.connect

    def connect(self, address, *args):
        if isinstance(address, tuple) and address[0] not in LOCAL:
            raise RuntimeError(f"a test tried to reach {address!r}; tests must stay offline")
        return real(self, address, *args)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
