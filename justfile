build:
    uv run hatch build -t custom

test:
    uv run coverage run -m pytest

coverage:
    uv run coverage report -m

coverage-html:
    uv run coverage html

publish:
    uv publish
