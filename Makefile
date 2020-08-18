PYTHON := python3
PYTEST := pytest
PIP := pip3
PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x --log-level=DEBUG
JULIA := julia --color=yes --project=src/julia
JULIA_SYSIMAGE := $(JULIA) --sysimage build/sysimage.so

all: docs test

build/sysimage.so: src/julia/Manifest.toml src/julia/Project.toml
	mkdir -p build
	$(JULIA) src/julia/sysimage.jl

develop:
	cd src/python && $(PYTHON) setup.py develop

docs:
	mkdocs build

install: install-python

install-python:
	cd src/python && $(PYTHON) setup.py install

install-julia:
	$(JULIA) -e "using Pkg; Pkg.instantiate()"

uninstall:
	$(PIP) uninstall miplearn

test: test-python test-julia

test-python:
	cd src/python && $(PYTEST) $(PYTEST_ARGS)

test-python-watch:
	cd src/python && pytest-watch -- $(PYTEST_ARGS)

test-julia: build/sysimage.so
	$(JULIA_SYSIMAGE) src/julia/test/runtests.jl

.PHONY: test test-python test-julia test-watch docs install
