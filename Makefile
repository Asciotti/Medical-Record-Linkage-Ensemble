all: venv

venv:
	virtualenv venv --python=python3.9
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install -r requirements_alec.txt
	touch venv/bin/activate
clean:
	rm -rf venv
	

jupyter_extentions:
	jupyter contrib nbextension install --user
	jupyter nbextensions_configurator enable --user

jupyter: jupyter_extentions
	jupyter notebook
