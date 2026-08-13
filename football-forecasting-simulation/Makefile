.PHONY: install test demo dashboard verify

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

demo:
	python -m football_lab.cli demo --output-dir reports/generated --simulations 2000

dashboard:
	streamlit run streamlit_app.py

verify:
	python -m compileall -q src tests streamlit_app.py
	python -m unittest discover -s tests -v
	python -m football_lab.cli demo --output-dir reports/generated --simulations 500
