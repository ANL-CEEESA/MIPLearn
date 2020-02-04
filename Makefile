PYTEST_ARGS := -W ignore::DeprecationWarning -vv -x

all: docs test

docs:
	mkdocs build

test:
	pytest $(PYTEST_ARGS)

test-watch:
	pytest-watch -- $(PYTEST_ARGS)

.PHONY: test test-watch docs
