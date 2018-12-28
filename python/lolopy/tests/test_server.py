from lolopy.loloserver import get_java_gateway
from py4j.java_gateway import java_import, JavaClass
from unittest import TestCase


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
