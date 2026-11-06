from __future__ import annotations

import pytest

from ttucs.taxonomy import UCSTaxonomy


@pytest.fixture(scope="session")
def taxonomy() -> UCSTaxonomy:
    return UCSTaxonomy.load()
