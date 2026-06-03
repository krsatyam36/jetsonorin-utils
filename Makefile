.PHONY: run test lint clean docker-build docker-run

run:
	python3 src/stream.py

test:
	python3 -m pytest tests/ -v --cov=src --cov-report=term

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage

docker-build:
	docker build -t jetson-stream .

docker-run:
	docker run --rm --device /dev/video0 -p 5000:5000 jetson-stream
