# Makefile for Water Body Segmentation Project

.PHONY: install run test docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	python app.py

test:
	pytest tests/

docker-build:
	docker build -t waterbody-segmentation-app .

docker-run:
	docker run -p 7860:7860 waterbody-segmentation-app

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf tests/__pycache__
