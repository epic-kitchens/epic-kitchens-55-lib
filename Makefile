.PHONY: all docs test compile upload_to_pypi typecheck clean format


LIBRARY_DIR := epic_kitchens
SRC_FILES := $(shell find epic_kitchens) 
SRC_FILES += setup.py
SYSTEM_TEST_DATASET_URL := "https://s3-eu-west-1.amazonaws.com/wp-research-public/epic/system_test_dataset.zip"
SYSTEM_TEST_DATASET_ETAG := $(shell cat tests/system_test_dataset.etag) 

all: test

docs:
	$(MAKE) html -C docs 

test: compile tests/dataset
	tox

tests/dataset: tests/system_test_dataset.zip
	mkdir -p "$@"
	unzip  -o "$<" -d "$@"
	touch "$@" # Update creation time of folder to prevent rule from rerunning

tests/system_test_dataset.zip: tests/system_test_dataset.etag
	if [ -f "$@" ]; then\
		curl "$(SYSTEM_TEST_DATASET_URL)" -o "$@" --header "If-None-Match: $(SYSTEM_TEST_DATASET_ETAG)";\
	else\
		curl "$(SYSTEM_TEST_DATASET_URL)" -o "$@";\
	fi


compile:
	python -m compileall $(LIBRARY_DIR) -j $$(nproc)

format:
	black epic_kitchens

flake8:
	flake8 epic_kitchens

dist: $(SRC_FILES) compile
	rm -rf dist
	python setup.py sdist

upload_to_pypi: dist
	twine upload dist/*

typecheck:
	mypy $(LIBRARY_DIR) --ignore-missing-imports

clean:
	rm -rf dist
