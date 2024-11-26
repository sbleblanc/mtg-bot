requirements.txt : pyproject.toml
	uv pip compile pyproject.toml --universal --output-file requirements.txt

install-deps:
	uv pip sync requirements.txt
	uv pip install -e .

test:
	pytest tests/