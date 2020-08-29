PYTHON      := python3
PYTEST      := pytest
PIP         := pip3
PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG

all: docs test

clean:
	rm -rf build

develop:
	$(PYTHON) setup.py develop

docs:
	mkdocs build

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py install

uninstall:
	$(PIP) uninstall miplearn

test:
	$(PYTEST) $(PYTEST_ARGS)

.PHONY: test test-watch docs install
