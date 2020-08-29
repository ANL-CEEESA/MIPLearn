PYTHON      := python3
PYTEST      := pytest
PIP         := pip3
PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG

all: docs test

develop:
	cd src/python && $(PYTHON) setup.py develop

docs:
	mkdocs build

install:
	cd src/python && $(PYTHON) setup.py install

uninstall:
	$(PIP) uninstall miplearn

test:
	cd src/python && $(PYTEST) $(PYTEST_ARGS)

.PHONY: test test-watch docs install
