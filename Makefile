PYTEST_ARGS := -W ignore::DeprecationWarning -vv

test:
	pytest $(PYTEST_ARGS)

test-watch:
	pytest-watch -- $(PYTEST_ARGS)

.PHONY: test test-watch
