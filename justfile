check:
    uvx pyrefly check

test:
    uv run coverage run -m pytest

coverage:
    uv run coverage report -m

coverage-html:
    uv run coverage html

build:
    uv run hatch build -t custom

publish:
    uv publish
