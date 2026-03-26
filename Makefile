# Makefile for Water Body Segmentation Project

.PHONY: install run test docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/

docker-build:
	docker build -t waterbody-segmentation-app .

docker-run:
	docker run -p 8501:8501 waterbody-segmentation-app

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf tests/__pycache__
