PYTHON      := python3
PYTEST      := pytest
PIP         := $(PYTHON) -m pip
MYPY        := $(PYTHON) -m mypy
PYTEST_ARGS := -W ignore::DeprecationWarning -vv --log-level=DEBUG
VERSION     := 0.2

all: docs test

clean:
	rm -rf build/* dist/*

develop:
	$(PYTHON) setup.py develop

dist:
	$(PYTHON) setup.py sdist bdist_wheel

dist-upload:
	$(PYTHON) -m twine upload dist/*

docs:
	rm -rf ../docs/$(VERSION) 
	cd docs; make clean; make dirhtml
	rsync -avP --delete-after docs/_build/dirhtml/ ../docs/$(VERSION)


install-deps:
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -i https://pypi.gurobi.com 'gurobipy>=9.5,<9.6'
	$(PIP) install --upgrade xpress
	$(PIP) install --upgrade -r requirements.txt

install:
	$(PYTHON) setup.py install

uninstall:
	$(PIP) uninstall miplearn

reformat:
	$(PYTHON) -m black .

test:
	rm -rf .mypy_cache
	$(MYPY) -p miplearn
	$(MYPY) -p tests
	$(MYPY) -p benchmark
	$(PYTEST) $(PYTEST_ARGS) 

.PHONY: test test-watch docs install dist
