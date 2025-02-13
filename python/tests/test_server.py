from lolopy.loloserver import get_java_gateway
from py4j.java_gateway import java_import, JavaClass
from unittest import TestCase
import os


class TestLoloGateway(TestCase):

    def test_launch(self):
        # Launch the gateway
        gate = get_java_gateway()

        # Make sure it runs by making a random number
        rnd = gate.jvm.java.util.Random()
        self.assertIsInstance(rnd.nextInt(), int)

        # Make sure importing Lolo works
        java_import(gate.jvm, "io.citrine.lolo.learners.*")
        self.assertIsInstance(gate.jvm.RandomForest, JavaClass)

        # Make sure requsting a gateway againt returns the same gateway
        gate2 = get_java_gateway()
        self.assertIs(gate, gate2)

        # Test getting a new gateway if needed
        gate3 = get_java_gateway(reuse=False)
        self.assertIsNot(gate, gate3)

        # Make the server using the package version of lolo
        gate4 = get_java_gateway(reuse=False, skip_devel_version=True)
        java_import(gate4.jvm, "io.citrine.lolo.learners.*")
        self.assertIsInstance(gate4.jvm.RandomForest, JavaClass)

    def test_memory(self):
        # Set an environmental variable (local for this test)
        os.environ['LOLOPY_JVM_MEMORY'] ='4g'

        with self.assertLogs("py4j.java_gateway", level='DEBUG') as cm:
            # Get a gateway
            get_java_gateway(reuse=False)

            # Make sure the memory amount appears in the logs
            self.assertIn('Xmx4g', '\n'.join(cm.output))

