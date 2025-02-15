"""Methods related to starting and stopping the Java Gateway"""
from pathlib import Path
from py4j.java_gateway import JavaGateway
from os import environ
import sys

# Directory where the lolo project root should be
_lolopy_dir = Path(__file__).parent
_lolo_root = _lolopy_dir.parent.parent

# Used for allowing multiple objects to use the same gateway
_lolopy_gateway = None


def _is_development_installation():
    """Check whether lolopy is in a folder with the rest of lolo"""

    # Look for the lolo scala source directory
    scala_dir = _lolo_root / 'src' / 'main' / 'scala' / 'io' / 'citrine' / 'lolo'
    return scala_dir.is_dir()


def find_lolo_jar(skip_devel_version=False):
    """Attempt to automatically find a jar file for Lolo

    Args:
        skip_devel_version (bool): Skip looking for the development version of lolo
    Returns:
        (string) Path to the Jar file
    """

    if not skip_devel_version and _is_development_installation():
        # Get the appropriate Jar
        jar_path = _lolo_root / 'target' / 'scala-2.13' / 'lolo-jar-with-dependencies.jar'
        if not jar_path.is_file():
            raise RuntimeError('Current version of lolo jar not found. Try re-building project with make')

    else:
        # Use the local installation
        jar_path = _lolopy_dir / 'jar' / 'lolo-jar-with-dependencies.jar'
        if not jar_path.is_file():
            raise RuntimeError(f'Lolo not found. Try reinstalling lolo from PyPi.')

    return jar_path


def get_java_gateway(reuse=True, skip_devel_version=False):
    """Get a JavaGateway with Lolo on the class path

    Args:
        reuse (bool): Whether to reuse an already-existing gateway
        skip_devel_version (bool): Whether to skip looking for the development version of lolopy
    Returns:
        (JavaGateway) A launched JavaGateway instance
    """

    global _lolopy_gateway

    # Set any default java options
    java_options = []  # No default options for now

    # Get an environmental variable set for the amount of heap memory
    if 'LOLOPY_JVM_MEMORY' in environ:
        java_options.append('-Xmx' + environ['LOLOPY_JVM_MEMORY'])

    # Make the gateway if none already active or user requests a fresh JVM
    if _lolopy_gateway is None or not reuse:
        lolo_path = find_lolo_jar(skip_devel_version)
        _lolopy_gateway = JavaGateway.launch_gateway(
            classpath=str(lolo_path.absolute()),
            javaopts=java_options,
            redirect_stdout=sys.stdout, die_on_exit=True)
    return _lolopy_gateway
