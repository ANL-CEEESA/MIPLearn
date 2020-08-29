PYTHON      := python3
PYTEST      := pytest
PIP         := pip3
PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG
VERSION     := `cat VERSION | sed 's/\.[0-9]*$$//'`

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

install:
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py install

uninstall:
	$(PIP) uninstall miplearn

test:
	$(PYTEST) $(PYTEST_ARGS)

.PHONY: test test-watch docs install
