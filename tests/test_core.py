"""The suite's own guard. The runtime's client, cache and retries are tested in jevkit-core."""

import socket

import pytest


def test_the_tests_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="tests must stay offline"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)  # refused by conftest before a packet leaves
