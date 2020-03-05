PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG

all: docs test

develop:
	cd src/python && python setup.py develop
	julia --project=src/julia -e "using Pkg; Pkg.instantiate()"

install:
	cd src/python && python setup.py install
	julia --project=src/julia -e "using Pkg; Pkg.instantiate()"

uninstall:
	pip uninstall miplearn

docs:
	mkdocs build

test:
	cd src/python && pytest $(PYTEST_ARGS)

test-watch:
	cd src/python && pytest-watch -- $(PYTEST_ARGS)

.PHONY: test test-watch docs
