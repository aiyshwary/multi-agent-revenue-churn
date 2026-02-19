.PHONY: build up down run docker-run test

build:
	# build a local docker image
	docker build -t multi-agent-architecture:local .

up:
	# build and run in background using docker-compose
	docker-compose up -d --build

down:
	# stop and remove containers
	docker-compose down

run:
	# run locally (no docker)
	python run_orchestrator.py --data data/sample_input.csv

docker-run:
	# run the local image and mount outputs/data
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/data:/app/data multi-agent-architecture:local

test:
	pytest -q
