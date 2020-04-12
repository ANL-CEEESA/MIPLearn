PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG
JULIA := julia --color=yes --project=src/julia

all: docs test

develop:
	cd src/python && python setup.py develop
	$(JULIA) -e "using Pkg; Pkg.instantiate()"

install:
	cd src/python && python setup.py install
	$(JULIA) -e "using Pkg; Pkg.instantiate()"

uninstall:
	pip uninstall miplearn

docs:
	mkdocs build

test: test-python test-julia

test-python:
	cd src/python && pytest $(PYTEST_ARGS)

test-julia:
	$(JULIA) -e 'using Pkg; Pkg.test("MIPLearn")'

test-watch:
	cd src/python && pytest-watch -- $(PYTEST_ARGS)

.PHONY: test test-python test-julia test-watch docs
