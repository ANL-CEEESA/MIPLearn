PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG
JULIA := julia --color=yes --project=src/julia --sysimage build/sysimage.so

all: docs test

build/sysimage.so: src/julia/Manifest.toml src/julia/Project.toml
	mkdir -p build
	julia --color=yes --project=src/julia src/julia/sysimage.jl

develop:
	cd src/python && python setup.py develop
	$(JULIA) -e "using Pkg; Pkg.instantiate()"

docs:
	mkdocs build

install:
	cd src/python && python setup.py install
	$(JULIA) -e "using Pkg; Pkg.instantiate()"

uninstall:
	pip uninstall miplearn

test: test-python test-julia

test-python:
	cd src/python && pytest $(PYTEST_ARGS)

test-python-watch:
	cd src/python && pytest-watch -- $(PYTEST_ARGS)

test-julia: build/sysimage.so
	$(JULIA) src/julia/test/runtests.jl

.PHONY: test test-python test-julia test-watch docs
