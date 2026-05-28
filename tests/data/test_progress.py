import pytest

from adrs.data.progress import inner_task, _progress_var, progress_context


@pytest.mark.asyncio
async def test_inner_task_no_op_when_contextvar_unset():
    """Outside progress_context, inner_task must be a no-op that still yields a usable handle."""
    assert _progress_var.get() is None

    async with inner_task("topic-x", total=10) as bar:
        bar.advance(3)
        bar.advance(7)
        assert bar.completed == 0  # no-op handle does not track
        assert bar.total is None


@pytest.mark.asyncio
async def test_progress_context_sets_and_resets_contextvar():
    """progress_context sets the contextvar inside the body and resets on exit."""
    assert _progress_var.get() is None
    async with progress_context(total_topics=2) as outer:
        assert _progress_var.get() is not None
        outer.advance(1)
        outer.advance(1)
        assert outer.completed == 2
        assert outer.total == 2
    assert _progress_var.get() is None


@pytest.mark.asyncio
async def test_inner_task_tracks_when_active():
    """Inside progress_context, inner_task records advances on its handle."""
    async with progress_context(total_topics=1):
        async with inner_task("topic-y", total=5) as bar:
            bar.advance(2)
            bar.advance(3)
            assert bar.completed == 5
            assert bar.total == 5


@pytest.mark.asyncio
async def test_inner_task_indeterminate():
    """total=None is allowed (block topics); advance still updates the handle."""
    async with progress_context(total_topics=1):
        async with inner_task("topic-z", total=None) as bar:
            bar.advance(123)
            assert bar.completed == 123
            assert bar.total is None
