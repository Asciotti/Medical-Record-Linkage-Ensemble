all: venv

venv:
	virtualenv venv --python=python3.9
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements_alec.txt
	touch venv/bin/activate
clean:
	rm -rf venv
