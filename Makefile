.PHONY: all docs test

LIBRARY_DIR:=epic_kitchens

all: test

docs:
	$(MAKE) html -C docs 

test:
	tox

typecheck:
	mypy $(LIBRARY_DIR) --ignore-missing-imports
