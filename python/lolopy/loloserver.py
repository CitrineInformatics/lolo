"""Methods related to starting and stopping the Java Gateway"""
from py4j.java_gateway import JavaGateway
from xml.etree import ElementTree
import os

# Directory where the lolo project root should be
_lolo_root = os.path.join(os.path.dirname(__file__), '..', '..')

# Used for allowing multiple objects to use the same gateway
_lolopy_gateway = None


def _is_development_installation():
    """Check whether lolopy is in a folder with the rest of lolo"""

    # Look for the lolo scala source directory
    return os.path.isdir(os.path.join(_lolo_root, 'src', 'main', 'scala', 'io', 'citrine', 'lolo'))


def find_lolo_jar(skip_devel_version=False):
    """Attempt to automatically find a jar file for Lolo

    Args:
        skip_devel_version (bool): Skip looking for the development version of lolo
    Returns:
        (string) Path to the Jar file
    """

    if not skip_devel_version and _is_development_installation():
        # Get the version of lolo from the pom.xml file
        pom = ElementTree.parse(os.path.join(_lolo_root, 'pom.xml'))
        lolo_version = pom.find('{http://maven.apache.org/POM/4.0.0}version').text

        # Get the appropriate Jar
        jar_path = os.path.join(_lolo_root, 'target', 'lolo-{}-jar-with-dependencies.jar'.format(lolo_version))
        if not os.path.isfile(jar_path):
            raise RuntimeError('Current version of lolo jar not found. Try re-building project with make')

    else:
        # Use the local installation
        jar_path = os.path.join(os.path.dirname(__file__), 'jar', 'lolo-jar-with-dependencies.jar')
        if not os.path.isfile(jar_path):
            raise RuntimeError('Lolo not found. Try reinstalling lolo from PyPi.')

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

    if _lolopy_gateway is None or not reuse:
        lolo_path = find_lolo_jar(skip_devel_version)
        _lolopy_gateway = JavaGateway.launch_gateway(classpath=os.path.pathsep.join([
            os.path.abspath(lolo_path)]), die_on_exit=True)
    return _lolopy_gateway
