import socket

import pytest


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    """Keep the core test suite independent of credentials and network access."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("FISHCREATE_ROOT", raising=False)

    def reject_network(*args, **kwargs):
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
