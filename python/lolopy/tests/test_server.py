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
