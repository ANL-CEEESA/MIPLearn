PYTEST_ARGS := -W ignore::DeprecationWarning --capture=no -vv

test:
	pytest $(PYTEST_ARGS)

test-watch:
	pytest-watch -- $(PYTEST_ARGS)

.PHONY: test test-watch
