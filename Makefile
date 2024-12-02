requirements.txt : pyproject.toml
	uv pip compile pyproject.toml --universal --output-file requirements.txt

install-deps:
	uv pip sync requirements.txt
	uv pip install -e .

serve-model:
	llama-server -m models/Phi-3.5-mini-instruct-Q6_K_L.gguf --port 8080 -ngl 33

test:
	pytest tests/