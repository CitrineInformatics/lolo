"""Methods related to starting and stopping the Java Gateway"""
from py4j.java_gateway import JavaGateway
from subprocess import STDOUT
import sys
import os

# Used for allowing multiple objects to use the same gateway
_lolopy_gateway = None

# TODO: Auto-detect the version of lolo? [Requires lolo installation (not compatible with PyPi) and/or internet connection to check Maven, and causes a maintenance burden]
_lolo_version = '1.0.3'


def find_lolo_jar():
    """Attempt to automatically find a jar file for Lolo

    Returns:
        (string) Path to the Jar file
    """

    # TODO: Make this not hardcoded -lw
    return os.path.join(os.path.dirname(__file__), '..', '..', 'target',
                        'lolo-{}-jar-with-dependencies.jar'.format(_lolo_version))


def get_java_gateway(reuse=True):
    """Get a JavaGateway with Lolo on the class path

    Args:
        reuse (bool): Whether to reuse an already-existing gateway
    Returns:
        (JavaGateway) A launched JavaGateway instance
    """

    global _lolopy_gateway

    if _lolopy_gateway is None or not reuse:
        lolo_path = find_lolo_jar()
        assert os.path.isfile(lolo_path), 'Lolo jar not found'

        _lolopy_gateway = JavaGateway.launch_gateway(classpath=os.path.pathsep.join([
            os.path.abspath(lolo_path)]), die_on_exit=True)
    return _lolopy_gateway
