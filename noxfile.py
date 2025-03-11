"""noxfile.py."""

import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
@nox.parametrize("resolution", ["highest", "lowest-direct"])
def tests(session: nox.Session, resolution: str) -> None:
    """Test session."""
    session.install("pytest~=8.3", "pytest-cov~=5.0")
    session.install(".", "--resolution", resolution)
    session.run("python", "-m", "pytest", *session.posargs)


@nox.session(default=False)
def coverage(session: nox.Session) -> None:
    """Coverage session."""
    tests(session, "highest")
    session.install("pytest-cov~=5.0")
    session.run("coverage", "html")
    session.run("coverage", "report")


@nox.session
def lint(session: nox.Session) -> None:
    """Lint session."""
    session.install("pre-commit~=3.8")
    session.run("pre-commit", "run", "--all-files", *session.posargs)
