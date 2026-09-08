import os

import pytest

from neoform.adapters.tinker import TinkerAdapter


@pytest.mark.skipif(
    not os.getenv("TINKER_API_KEY") or os.getenv("NEOFORM_LIVE_TEST") != "1",
    reason="Live test requires explicit key and NEOFORM_LIVE_TEST=1 opt-in",
)
@pytest.mark.asyncio
async def test_live_tinker_connection():
    result = await TinkerAdapter().doctor()
    assert result["connected"] is True
    assert result["provider"] == "tinker"
