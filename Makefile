MAKEFLAGS += -j2
OS := $(shell uname;)
SELINUXOPT ?= $(shell test -x /usr/sbin/selinuxenabled && selinuxenabled && echo -Z)
PREFIX ?= /usr/local
BINDIR ?= ${PREFIX}/bin
SHAREDIR ?= ${PREFIX}/share
PYTHON ?= $(shell command -v python3 python|head -n1)
DESTDIR ?= /
PATH := $(PATH):$(HOME)/.local/bin
MYPIP ?= pip
IMAGE ?= ramalama
PROJECT_DIR ?= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
EXCLUDE_DIRS ?= .venv venv .tox build
EXCLUDE_OPTS ?= $(addprefix --exclude-dir=,$(EXCLUDE_DIRS))
PYTHON_SCRIPTS ?= $(shell grep -lEr "^\#\!\s*/usr/bin/(env +)?python(3)?(\s|$$)" $(EXCLUDE_OPTS) $(PROJECT_DIR) || true)
RUFF_TARGETS ?= ramalama scripts test bin/ramalama
E2E_IMAGE ?= localhost/e2e:latest

# Set COVERAGE=1 on any test target to measure code coverage, e.g.
# `make COVERAGE=1 e2e-tests`. Each run leaves a combined .coverage data file in
# the project directory and an XML report under coverage/; use
# `make coverage-combine` to merge data files from several runs into a single
# report.
#
# COVERAGE_FILE must be absolute: the e2e tests chdir into a temporary workspace
# which they delete afterwards, and subprocess data files are written relative to
# the current directory. COVERAGE_PROCESS_START activates the .pth hook tox
# installs into the test environment (see commands_pre in pyproject.toml).
ifdef COVERAGE
COV_OPTS := --cov --cov-report=term --cov-report=xml
export COVERAGE_PROCESS_START := $(PROJECT_DIR)/pyproject.toml
export COVERAGE_FILE := $(PROJECT_DIR)/.coverage
endif
COV_POSARGS := $(if $(COV_OPTS),-- $(COV_OPTS))

default: help

help:
	@echo "Build Container Image"
	@echo
	@echo "  - make build"
	@echo "  - make build IMAGE=ramalama"
	@echo "  - make multi-arch"
	@echo "  - make multi-arch IMAGE=ramalama"
	@echo "  Build using build cache, for development only"
	@echo "  - make build IMAGE=ramalama CACHE=-C"
	@echo
	@echo "Build docs"
	@echo
	@echo "  - make docs"
	@echo
	@echo "Install ramalama"
	@echo
	@echo "  - make install"
	@echo
	@echo "Test ramalama"
	@echo
	@echo "  - make test"
	@echo
	@echo "Clean the repository"
	@echo
	@echo "  - make clean"
	@echo

.PHONY: install-uv
install-uv:
	./install-uv.sh

.PHONY: install-requirements
install-requirements:
	${MYPIP} install ".[dev]"

.PHONY: install-completions
install-completions: completions
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/bash-completion/completions
	install ${SELINUXOPT} -m 644 completions/bash-completion/completions/ramalama \
		$(DESTDIR)${SHAREDIR}/bash-completion/completions/ramalama
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/fish/vendor_completions.d
	install ${SELINUXOPT} -m 644 completions/fish/vendor_completions.d/ramalama.fish \
		$(DESTDIR)${SHAREDIR}/fish/vendor_completions.d/ramalama.fish
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)${SHAREDIR}/zsh/site-functions
	install ${SELINUXOPT} -m 644 completions/zsh/site-functions/_ramalama \
		$(DESTDIR)${SHAREDIR}/zsh/site-functions/_ramalama

.PHONY: install-shortnames
install-shortnames:
	install ${SELINUXOPT} -d -m 755 $(DESTDIR)$(SHAREDIR)/ramalama
	install ${SELINUXOPT} -m 644 shortnames/shortnames.conf \
		$(DESTDIR)$(SHAREDIR)/ramalama

.PHONY: completions
completions:
	mkdir -p completions/bash-completion/completions
	register-python-argcomplete --shell bash ramalama > completions/bash-completion/completions/ramalama

	mkdir -p completions/fish/vendor_completions.d
	register-python-argcomplete --shell fish ramalama > completions/fish/vendor_completions.d/ramalama.fish

	mkdir -p completions/zsh/site-functions
	-register-python-argcomplete --shell zsh ramalama > completions/zsh/site-functions/_ramalama

.PHONY: install
install: docs completions
	RAMALAMA_VERSION=$(RAMALAMA_VERSION) \
	${MYPIP} install . --no-deps --root $(DESTDIR) --prefix ${PREFIX}

.PHONY: build
build:
	./container_build.sh ${CACHE} build $(IMAGE) -v "$(VERSION)"

.PHONY: build-rm
build-rm:
	./container_build.sh ${CACHE} -r build $(IMAGE) -v "$(VERSION)"

.PHONY: build_multi_arch
build_multi_arch:
	./container_build.sh ${CACHE} multi-arch $(IMAGE) -v "$(VERSION)"

.PHONY: install-docs
install-docs: docs
	make -C docs install

.PHONY: docs docs-manpages docsite-docs
docs: docs-manpages docsite-docs

docs-manpages:
	$(MAKE) -C docs

# Preprocess *.md.in into *.md only (no go-md2man). Used by man-check in CI.
.PHONY: docs-manpages-md
docs-manpages-md:
	$(MAKE) -C docs manpages-md

docsite-docs:
	$(MAKE) -C docsite convert

.PHONY: lint
lint:
	! git grep -n -- '#!/usr/bin/python3' -- ':!Makefile'
	ruff check $(RUFF_TARGETS)
	shellcheck *.sh */*.sh */*/*.sh

.PHONY: check-format
check-format:
	ruff check --select I $(RUFF_TARGETS)
	ruff format --check $(RUFF_TARGETS)
	$(PYTHON) ramalama/shortnames.py --check shortnames/shortnames.conf

.PHONY: format
format:
	ruff check --select I --fix $(RUFF_TARGETS)
	ruff format $(RUFF_TARGETS)
	$(PYTHON) ramalama/shortnames.py shortnames/shortnames.conf

.PHONY: codespell
codespell:
	codespell $(PROJECT_DIR) $(PYTHON_SCRIPTS)

.PHONY: man-check
man-check: docs-manpages-md
	@if ! git diff --quiet -- docs/ || \
	    [ -n "$$(git ls-files --others --exclude-standard docs/)" ]; then \
		echo "ERROR: generated man-page markdown is out of date."; \
		echo "Run 'make docs-manpages-md' and commit the results."; \
		git diff --name-only -- docs/; \
		git ls-files --others --exclude-standard docs/; \
		exit 1; \
	fi
ifeq ($(OS),Linux)
	hack/man-page-checker
	hack/xref-helpmsgs-manpages
endif

.PHONY: type-check
type-check:
	mypy --check-untyped-defs $(addprefix --exclude=,$(EXCLUDE_DIRS)) --exclude test $(PROJECT_DIR)

.PHONY: validate
validate: codespell lint check-format man-check type-check

.PHONY: pypi-build
pypi-build:   clean
	make docs
	python3 -m build --sdist
	python3 -m build --wheel

.PHONY: pypi
pypi: pypi-build
	python3 -m twine upload dist/*

.PHONY: e2e-image
e2e-image:
	podman inspect $(E2E_IMAGE) &> /dev/null || \
		podman build -t $(E2E_IMAGE) -f container-images/e2e/Containerfile .

e2e-tests-in-container slow-tests-in-container: extra-opts = --security-opt unmask=/proc/* --device /dev/net/tun

%-in-container: e2e-image
	podman run --rm \
		--userns=keep-id:size=200000 \
		--security-opt label=disable \
		--security-opt=mask=/sys/bus/pci/drivers/i915 \
		$(extra-opts) \
		-v /tmp \
		-v $(CURDIR):/src \
		$(E2E_IMAGE) make $*

.PHONY: ci
ci:
	test/ci.sh

.PHONY: requires-tox
requires-tox:
	@command -v tox >/dev/null 2>&1 || ${MYPIP} install tox

.PHONY: unit-tests
unit-tests: requires-tox
	tox $(COV_POSARGS)

.PHONY: unit-tests-verbose
unit-tests-verbose: requires-tox
	tox -- --full-trace --capture=tee-sys $(COV_OPTS)

.PHONY: cov-tests
cov-tests:
	$(MAKE) COVERAGE=1 unit-tests

.PHONY: detailed-cov-tests
detailed-cov-tests: requires-tox
	tox -e coverage

.PHONY: e2e-tests
e2e-tests: requires-tox
	tox -q -e e2e $(COV_POSARGS)

.PHONY: e2e-tests-nocontainer
e2e-tests-nocontainer: requires-tox
	tox -q -e e2e -- --no-container $(COV_OPTS)

.PHONY: e2e-tests-docker
e2e-tests-docker: requires-tox
	tox -q -e e2e -- --container-engine=docker $(COV_OPTS)

.PHONY: slow-tests
slow-tests: requires-tox
	tox -q -e slow $(COV_POSARGS)

.PHONY: slow-tests-docker
slow-tests-docker: requires-tox
	tox -q -e slow -- --container-engine=docker $(COV_OPTS)

# Merge coverage data files produced by several test runs (or downloaded from
# several CI jobs) into a single report. COVERAGE_INPUTS may name files or
# directories; directories are searched for .coverage.* data files.
COVERAGE_INPUTS ?=
.PHONY: coverage-combine
coverage-combine:
	coverage combine --keep $(COVERAGE_INPUTS)
	coverage report --precision=2 --skip-covered
	coverage html
	coverage xml

.PHONY: end-to-end-tests
end-to-end-tests: validate e2e-tests e2e-tests-nocontainer slow-tests ci
	make clean
	hack/tree_status.sh

.PHONY: test
test: tests

.PHONY: tests
tests: unit-tests end-to-end-tests

.PHONY: rag-requirements
rag-requirements:
	touch container-images/common/*.in
	make -C container-images/common tools-requirements requirements-rag.txt

.PHONY: clean
clean:
	make -C docs clean
	make -C docsite clean clean-generated
	find . -depth -print0 | git check-ignore --stdin -z | xargs -0 rm -rf
