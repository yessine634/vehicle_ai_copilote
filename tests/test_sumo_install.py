"""Verify that the active Python environment can access a Windows SUMO installation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess


def require_python_package(package_name: str) -> None:
    """Raise a helpful error when a required Python package is unavailable."""

    if importlib.util.find_spec(package_name) is None:
        raise RuntimeError(
            f"Python package '{package_name}' is missing from the active environment. "
            "Activate llm-env and run: pip install traci sumolib"
        )


def main() -> None:
    """Run the SUMO environment and Python integration checks."""

    print("1. Checking SUMO_HOME...")
    sumo_home = os.environ.get("SUMO_HOME")
    print("SUMO_HOME:", sumo_home)
    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME is missing. The Windows SUMO installer should set it "
            "automatically; reopen the terminal after installation and try again."
        )
    if not Path(sumo_home).is_dir():
        raise RuntimeError(f"SUMO_HOME does not point to a directory: {sumo_home}")

    print("\n2. Checking TraCI before importing it...")
    require_python_package("traci")
    import traci  # noqa: F401

    print("TraCI import successful")

    print("\n3. Checking sumolib before importing it...")
    require_python_package("sumolib")
    import sumolib  # noqa: F401

    print("sumolib import successful")

    print("\n4. Looking for the SUMO executable...")
    sumo_executable = shutil.which("sumo")
    print("SUMO executable:", sumo_executable)
    if not sumo_executable:
        raise RuntimeError(
            "Python cannot find 'sumo'. Reopen the terminal, then confirm that "
            "%SUMO_HOME%\\bin is included in PATH."
        )

    print("\n5. Running sumo --version...")
    result = subprocess.run(
        [sumo_executable, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version_output = result.stdout.strip() or result.stderr.strip()
    print(version_output)

    print("\n6. Looking for SUMO GUI without opening it...")
    sumo_gui_executable = shutil.which("sumo-gui")
    print("SUMO GUI:", sumo_gui_executable)
    if not sumo_gui_executable:
        raise RuntimeError(
            "Python cannot find 'sumo-gui'. Confirm that %SUMO_HOME%\\bin is "
            "included in PATH, then reopen the terminal."
        )

    print("\nInstallation summary")
    print("SUMO_HOME: OK")
    print("SUMO executable: OK")
    print("SUMO GUI: OK")
    print("TraCI: OK")
    print("sumolib: OK")
    print("\nSUMO Python environment is ready.")


if __name__ == "__main__":
    main()
