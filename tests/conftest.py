import socket

import pytest


@pytest.fixture(autouse=True)
def no_network_and_no_keys(monkeypatch):
    def refuse(*arguments, **keywords):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
