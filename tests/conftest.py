import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeEmbedder, FakeLLM, Reply  # noqa: E402,F401


@pytest.fixture
def llm_servers():
    """Factory for fake servers; all are shut down after the test."""
    made: list[FakeLLM] = []

    def make(responder=None, **kw) -> FakeLLM:
        s = FakeLLM(responder, **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder()
