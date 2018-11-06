"""Methods related to starting and stopping the Java Gateway"""
from py4j.java_gateway import JavaGateway
import os

def _find_lolo_jar():
    """Attempt to automatically find a jar file for Lolo

    Returns:
        (string) Path to the Jar file
    """

    # TODO: Make this not hardcoded -lw
    return os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'lolo-1.0.2.jar')


def get_java_gateway(reuse=True):
    """Get a JavaGateway with Lolo on the class path

    Args:
        reuse (bool): Whether to reuse an already-existing gateway (TBD)
    Returns:
        (JavaGateway) A launched JavaGateway instance
    """
    # TODO: Implement a way to prevent having to launch a new JVM every time
    lolo_path = _find_lolo_jar()

    # TODO: Find a way to get the path to scala (might just detect if on Windows vs Linux
    _gateway = JavaGateway.launch_gateway(classpath=os.path.pathsep.join([
        os.path.abspath(lolo_path)]), die_on_exit=True)
    return _gateway

JavaGateway()
