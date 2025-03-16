from pathlib import Path

import nox

nox.options.reuse_existing_virtualenvs = True  # Reuse virtual environments
nox.options.sessions = ["precommit"]


def get_paths(session):
    package_path = Path(session.bin).parent.parent.parent
    return {
        "all": [
            package_path / "building_sci",
            package_path / "tests",
            package_path / "scripts",
        ],
        "module": [
            package_path / "building_sci",
            package_path / "scripts",
        ],
    }


#
# Sessions
#


@nox.session()
def pylint(session):
    paths = get_paths(session)
    session.run("poetry", "run", "pylint", *paths["module"], external=True)


@nox.session()
def flake8(session):
    paths = get_paths(session)
    session.run("poetry", "run", "flake8", *paths["all"], external=True)


@nox.session()
def docformatter(session):
    paths = get_paths(session)
    session.run(
        "poetry",
        "run",
        "docformatter",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )


@nox.session()
def check(session):
    paths = get_paths(session)
    session.run("poetry", "run", "black", "--check", *paths["all"], external=True)
    session.run("poetry", "run", "isort", *paths["all"], "--check", external=True)
    session.run("poetry", "run", "flynt", *paths["all"], external=True)
    session.run(
        "poetry",
        "run",
        "docformatter",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )
    session.run("poetry", "run", "flake8", *paths["all"], external=True)
    session.run("poetry", "run", "pylint", *paths["module"], external=True)


@nox.session()
def fix(session):
    paths = get_paths(session)
    session.run("poetry", "run", "black", *paths["all"], external=True)
    session.run("poetry", "run", "isort", *paths["all"], external=True)
    session.run("poetry", "run", "flynt", *paths["all"], external=True)
    session.run(
        "poetry",
        "run",
        "docformatter",
        "--in-place",
        "--config",
        f"{paths['all'][0].parent}/pyproject.toml",
        *paths["all"],
        external=True,
    )


@nox.session()
def precommit(session):
    session.run("poetry", "run", "pre-commit", "run", "--all-files", external=True)


@nox.session()
def black(session):
    paths = get_paths(session)
    session.run("poetry", "run", "black", "--check", *paths["all"], external=True)


@nox.session()
def isort(session):
    paths = get_paths(session)
    session.run("poetry", "run", "isort", *paths["all"], "--check", external=True)


@nox.session()
def flynt(session):
    paths = get_paths(session)
    session.run("poetry", "run", "flynt", *paths["all"], external=True)


@nox.session()
def test(session):
    session.run("poetry", "run", "pytest", external=True)


@nox.session()
def test_custom(session):
    session.run(
        "poetry", "run", "pytest", external=True, *session.posargs
    )  # Pass additional arguments directly to pytest


@nox.session()
def test_nb(session):
    session.run(
        "poetry",
        "run",
        "pytest",
        "--nbval",
        "tests/test_notebooks/",
        "--nbval-sanitize-with=tests/test_notebooks/sanitize_file.cfg",
        external=True,
    )
