.PHONY: all docs test

LIBRARY_DIR:=epic_kitchens

all: test

docs:
	$(MAKE) html -C docs 

test:
	tox

package:
	rm -rf dist
	python setup.py sdist

upload_to_pypi: package
	twine upload dist/*

typecheck:
	mypy $(LIBRARY_DIR) --ignore-missing-imports
