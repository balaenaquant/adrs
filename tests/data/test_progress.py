import pytest

from adrs.data.progress import inner_task, _progress_var


@pytest.mark.asyncio
async def test_inner_task_no_op_when_contextvar_unset():
    """Outside progress_context, inner_task must be a no-op that still yields a usable handle."""
    assert _progress_var.get() is None

    async with inner_task("topic-x", total=10) as bar:
        bar.advance(3)
        bar.advance(7)
        assert bar.completed == 0  # no-op handle does not track
        assert bar.total is None
