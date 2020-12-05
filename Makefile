PYTHON      := python3
PYTEST      := pytest
PIP         := $(PYTHON) -m pip
PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG
VERSION     := 0.2

all: docs test

clean:
	rm -rf build

develop:
	$(PYTHON) setup.py develop

dist:
	$(PYTHON) setup.py sdist bdist_wheel

dist-upload:
	$(PYTHON) -m twine upload dist/*

docs:
	mkdocs build -d ../docs/$(VERSION)/

docs-dev:
	mkdocs build -d ../docs/dev/

install-deps:
	$(PIP) install -i https://pypi.gurobi.com gurobipy
	$(PIP) install -r requirements.txt

install:
	$(PYTHON) setup.py install

uninstall:
	$(PIP) uninstall miplearn

reformat:
	$(PYTHON) -m black miplearn

test:
	$(PYTEST) $(PYTEST_ARGS)

.PHONY: test test-watch docs install
