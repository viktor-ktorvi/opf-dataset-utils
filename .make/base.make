########################################################################################
# DO NOT MODIFY!!!
# If necessary, override the corresponding variable and/or target, or create new ones
# in one of the following files, depending on the nature of the override :
#
# Makefile.variables, Makefile.targets or Makefile.private`,
#
# The only valid reason to modify this file is to fix a bug or to add new
# files to include.
#
# Please report bugs to francis.pelletier@mila.quebec
########################################################################################

# Basic variables
PROJECT_PATH := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
SHELL := /usr/bin/env bash
BUMP_TOOL := bump-my-version
APP_VERSION := 0.0.0
DOCKER_COMPOSE ?= docker compose
POETRY_VERSION := 1.8.2
AUTO_INSTALL ?=

# Conda variables
# CONDA_TOOL can be overridden in Makefile.private file
CONDA_TOOL := conda
CONDA_ENVIRONMENT ?=

# Colors
_SECTION := \033[1m\033[34m
_TARGET  := \033[36m
_NORMAL  := \033[0m

.DEFAULT_GOAL := help
## -- Informative targets ------------------------------------------------------------------------------------------- ##

.PHONY: all
all: help

# Auto documented help targets & sections from comments
#	detects lines marked by double #, then applies the corresponding target/section markup
#   target comments must be defined after their dependencies (if any)
#	section comments must have at least a double dash (-)
#
# 	Original Reference:
#		https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# 	Formats:
#		https://misc.flogisoft.com/bash/tip_colors_and_formatting
#
#	As well as influenced by it's implementation in the Weaver Project
#		https://github.com/crim-ca/weaver/tree/master

.PHONY: help
# note: use "\#\#" to escape results that would self-match in this target's search definition
help: ## print this help message (default)
	@echo ""
	@echo "Please use 'make <target>' where <target> is one of below options."
	@echo ""
	@for makefile in $(MAKEFILE_LIST); do \
        grep -E '\#\#.*$$' "$(PROJECT_PATH)/$${makefile}" | \
            awk 'BEGIN {FS = "(:|\-\-\-)+.*\#\# "}; \
            	/\--/ {printf "$(_SECTION)%s$(_NORMAL)\n", $$1;} \
				/:/  {printf "    $(_TARGET)%-24s$(_NORMAL) %s\n", $$1, $$2} ' 2>/dev/null ; \
    done

.PHONY: targets
targets: help

.PHONY: version
version: ## display current version
	@echo "version: $(APP_VERSION)"

## -- Conda targets ------------------------------------------------------------------------------------------------- ##

.PHONY: conda-install
conda-install: ## Install Conda on your local machine
	@echo "Looking for [$(CONDA_TOOL)]..."; \
	$(CONDA_TOOL) --version; \
	if [ $$? != "0" ]; then \
		echo " "; \
		echo "Your defined Conda tool [$(CONDA_TOOL)] has not been found."; \
		echo " "; \
		echo "If you know you already have [$(CONDA_TOOL)] or some other Conda tool installed,"; \
		echo "Check your [CONDA_TOOL] variable in the Makefile.private for typos."; \
		echo " "; \
		echo "If your conda tool has not been initiated through your .bashrc file,"; \
		echo "consider using the full path to its executable instead when"; \
		echo "defining your [CONDA_TOOL] variable"; \
		echo " "; \
		echo "If in doubt, don't install Conda and manually create and activate"; \
		echo "your own Python environment."; \
		echo " "; \
		echo -n "Would you like to install Miniconda ? [y/N]: "; \
		read ans; \
		case $$ans in \
			[Yy]*) \
				echo "Fetching and installing miniconda"; \
				echo " "; \
				wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
    			bash ~/miniconda.sh -b -p ${HOME}/.conda; \
    			export PATH=${HOME}/.conda/bin:$PATH; \
    			conda init; \
				/usr/bin/rm ~/miniconda.sh; \
				;; \
			*) \
				echo "Skipping installation."; \
				echo " "; \
				;; \
		esac; \
	else \
		echo "Conda tool [$(CONDA_TOOL)] has been found, skipping installation"; \
	fi;

.PHONY: conda-create-env
conda-create-env: conda-install ## Create a local Conda environment based on `environment.yml` file
	@$(CONDA_TOOL) env create -y -f environment.yml

.PHONY: conda-env-info
conda-env-info: ## Print information about active Conda environment using <CONDA_TOOL>
	@$(CONDA_TOOL) info

.PHONY: _conda-poetry-install
_conda-poetry-install:
	$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) $(CONDA_TOOL) install -y poetry==$(POETRY_VERSION); \



.PHONY:conda-poetry-install
conda-poetry-install: ## Install Poetry in currently active Conda environment. Will fail if Conda is not found
	@poetry --version; \
    	if [ $$? != "0" ]; then \
			echo "Poetry not found, proceeding to install Poetry..."; \
			echo "Looking for [$(CONDA_TOOL)]...";\
			$(CONDA_TOOL) --version; \
			if [ $$? != "0" ]; then \
				echo "$(CONDA_TOOL) not found; Poetry will not be installed"; \
			else \
				echo "Installing Poetry with Conda in [$(CONDA_ENVIRONMENT)] environment"; \
				make -s _conda-poetry-install; \
			fi; \
		fi;

.PHONY: conda-poetry-uninstall
conda-poetry-uninstall: ## Uninstall Poetry located in currently active Conda environment
	$(CONDA_TOOL) run -n $(CONDA_ENVIRONMENT) $(CONDA_TOOL) remove poetry

.PHONY: conda-clean-env
conda-clean-env: ## Completely removes local project's Conda environment
	$(CONDA_TOOL) env remove -n $(CONDA_ENVIRONMENT)

## -- Poetry targets ------------------------------------------------------------------------------------------------ ##

.PHONY: poetry-install-auto
poetry-install-auto: ## Install Poetry in activated Conda environment, or with pipx if Conda not found
	@poetry --version; \
    	if [ $$? != "0" ]; then \
			echo "Poetry not found, proceeding to install Poetry..."; \
			echo "Looking for [$(CONDA_TOOL)]...";\
			$(CONDA_TOOL) --version; \
            if [ $$? != "0" ]; then \
				echo "$(CONDA_TOOL) not found, trying with pipx"; \
				pipx --version; \
				if [ $$? != "0" ]; then \
					echo "pipx not found; installing pipx"; \
					pip install --user pipx; \
					pipx ensurepath; \
				fi; \
					pipx install poetry==$(POETRY_VERSION); \
				else \
					echo "Installing poetry with Conda"; \
					make -s _conda-poetry-install; \
				fi; \
		fi;

.PHONY: poetry-install
poetry-install: ## Install standalone Poetry using pipx and create Poetry env. Will install pipx if not found
	@echo "Looking for Poetry version...";\
	poetry --version; \
    	if [ $$? != "0" ]; then \
    	  	if [ "$(AUTO_INSTALL)" = "true" ]; then \
				ans="y";\
			else \
    	  		echo""; \
    	  		echo -n "Would you like to install Poetry using pipx? (This will also install pipx if needed) [y/N]: "; \
				read ans; \
			fi; \
			case $$ans in \
				[Yy]*) \
					pipx --version; \
					if [ $$? != "0" ]; then \
						echo "pipx not found; installing pipx"; \
						pip install --user pipx; \
						pipx ensurepath; \
					fi; \
						echo "Installing Poetry"; \
						pipx install poetry==$(POETRY_VERSION); \
						make -s poetry-create-env; \
					;; \
				*) \
					echo "Skipping installation."; \
					echo " "; \
					;; \
			esac; \
		fi;

.PHONY: poetry-env-info
poetry-env-info: ## Information about the currently active environment used by Poetry
	@poetry env info

.PHONY: poetry-create-env
poetry-create-env: ## Create a Poetry managed environment for the project (Outside of Conda environment).
	@echo "Creating Poetry environment that will use Python $(PYTHON_VERSION)"; \
	poetry env use $(PYTHON_VERSION); \
	poetry env info
	@echo""
	@echo "This environment can be accessed either by using the <poetry run YOUR COMMAND>"
	@echo "command, or activated with the <poetry shell> command."
	@echo""
	@echo "Use <poetry --help> and <poetry list> for more information"
	@echo""

.PHONY: poetry-remove-env
poetry-remove-env: ## Remove current project's Poetry managed environment.
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans_env="y";\
		env_path=$$(poetry env info -p); \
		env_name=$$(basename $$env_path); \
	else \
		echo""; \
		env_path=$$(poetry env info -p); \
		if [[ "$$env_path" != "" ]]; then \
			echo "The following environment has been found for this project: "; \
			env_name=$$(basename $$env_path); \
			echo""; \
			echo "Env name : $$env_name"; \
			echo "PATH     : $$env_path"; \
			echo""; \
			echo "If the active environment listed above is a Conda environment,"; \
			echo "Choosing to delete it will have no effect; use the target <make conda-clean-env>"; \
			echo""; \
			echo -n "Would you like delete the environment listed above? [y/N]: "; \
			read ans_env; \
		else \
		  env_name="None"; \
		  env_path="None"; \
  		fi; \
	fi; \
	if [[ $$env_name != "None" ]]; then \
		case $$ans_env in \
			[Yy]*) \
				poetry env remove $$env_name || echo "No environment was removed"; \
				;; \
			*) \
				echo "No environment was found/provided - skipping environment deletion"; \
				;;\
		esac; \
	fi; \


.PHONY: poetry-uninstall-pipx
poetry-uninstall-pipx: poetry-remove-env ## Uninstall pipx-installed Poetry, the created env and pipx
	@if [ "$(AUTO_INSTALL)" = "true" ]; then \
		ans="y";\
	else \
		echo""; \
		echo -n "Would you like to uninstall pipx-installed Poetry and pipx? [y/N]: "; \
		read ans; \
	fi; \
	case $$ans in \
		[Yy]*) \
			pipx uninstall poetry; \
			pip uninstall -y pipx; \
			;; \
		*) \
			echo "Skipping uninstallation."; \
			echo " "; \
			;; \
	esac; \

## -- Install targets (All install targets will install Poetry if not found using `make poetry-install-auto`)-------- ##

.PHONY: install
install: install-precommit ## Install the application package, developer dependencies and pre-commit hook

.PHONY: install-precommit
install-precommit: install-dev## Install the pre-commit hooks (also installs developer dependencies)
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "Pre-commit hook found"; \
	else \
	  	echo "Pre-commit hook not found, proceeding to configure it"; \
		poetry run pre-commit install; \
	fi;

.PHONY: install-dev
install-dev: poetry-install-auto ## Install the application along with developer dependencies
	@poetry install --with dev

.PHONY: install-with-lab
install-with-lab: poetry-install-auto ## Install the application and it's dev dependencies, including Jupyter Lab
	@poetry install --with dev --with lab


.PHONY: install-package
install-package: poetry-install-auto ## Install the application package only
	@poetry install

## -- Versioning targets ------------------------------------------------------------------------------------------- ##

# Use the "dry" target for a dry-run version bump, ex.
# make bump-major dry
BUMP_ARGS ?= --verbose
ifeq ($(filter dry, $(MAKECMDGOALS)), dry)
	BUMP_ARGS := $(BUMP_ARGS) --dry-run --allow-dirty
endif

.PHONY: bump-major
bump-major: ## Bump application major version  <X.0.0>
	$(BUMP_TOOL) $(BUMP_ARGS) bump major

.PHONY: bump-minor
bump-minor: ## Bump application minor version  <0.X.0>
	$(BUMP_TOOL) $(BUMP_ARGS) bump minor

.PHONY: bump-patch
bump-patch: ## Bump application patch version  <0.0.X>
	$(BUMP_TOOL) $(BUMP_ARGS) bump patch

## -- Docker targets ------------------------------------------------------------------------------------------------ ##

## -- Apptainer/Singularity targets --------------------------------------------------------------------------------- ##

## -- Linting targets ----------------------------------------------------------------------------------------------- ##

.PHONY: check-lint
check-lint: ## Check code linting (black, isort, flake8 and pylint)
	poetry run tox -e black,isort,flake8,pylint

.PHONY: check-pylint
check-pylint: ## Check code with pylint
	poetry run tox -e pylint

.PHONY: fix-lint
fix-lint: ## Fix code linting (black, isort, flynt)
	poetry run tox -e fix

.PHONY: precommit
precommit: ## Run Pre-commit on all files manually
	poetry run tox -e precommit

## -- Tests targets ------------------------------------------------------------------------------------------------- ##

.PHONY: test
test: ## Run all tests
	poetry run tox -e test

TEST_ARGS ?=
MARKER_TEST_ARGS = -m "$(TEST_ARGS)"
SPECIFIC_TEST_ARGS = -k "$(TEST_ARGS)"
CUSTOM_TEST_ARGS = "$(TEST_ARGS)"

.PHONY: test-marker
test-marker: ## Run tests using pytest markers. Ex. make test-tag TEST_ARGS="<marker>"
	@if [ -n "$(TEST_ARGS)" ]; then \
		poetry run tox -e test-custom -- -- $(MARKER_TEST_ARGS); \
	else \
		echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-marker TEST_ARGS="<marker>"' ; \
	  	echo "" ; \
    fi
.PHONY: test-specific
test-specific: ## Run specific tests using the -k option. Ex. make test-specific TEST_ARGS="<name-of-test>"
	@if [ -n "$(TEST_ARGS)" ]; then \
  		poetry run tox -e test-custom -- -- $(SPECIFIC_TEST_ARGS); \
	else \
		echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-specific TEST_ARGS="<name-of-the-test>"' ; \
	  	echo "" ; \
    fi

.PHONY: test-custom
test-custom: ## Run tests with custom args. Ex. make test-custom TEST_ARGS="-m 'not offline'"
	@if [ -n "$(TEST_ARGS)" ]; then \
  		poetry run tox -e test-custom -- -- $(CUSTOM_TEST_ARGS); \
	else \
	  	echo "" ; \
    	echo 'ERROR : Variable TEST_ARGS has not been set, please rerun the command like so :' ; \
	  	echo "" ; \
    	echo '            make test-custom TEST_ARGS="<custom-args>"' ; \
	  	echo "" ; \
    fi