.PHONY: all docs test

LIBRARY_DIR:=epic_kitchens

all: test

docs:
	$(MAKE) html -C docs 

test:
	tox

dist:
	rm -rf dist
	python setup.py sdist

upload_to_pypi: dist
	twine upload dist/*

typecheck:
	mypy $(LIBRARY_DIR) --ignore-missing-imports

clean:
	rm -rf dist
