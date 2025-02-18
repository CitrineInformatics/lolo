from lolopy import loloserver
from lolopy.loloserver import get_java_gateway, find_lolo_jar
from py4j.java_gateway import java_import, JavaClass

from pytest import raises

class TestLoloGateway:
    def test_launch(self):
        # Launch the gateway
        gate = get_java_gateway()

        # Make sure it runs by making a random number
        rnd = gate.jvm.java.util.Random()
        assert isinstance(rnd.nextInt(), int)

        # Make sure importing Lolo works
        java_import(gate.jvm, "io.citrine.lolo.learners.*")
        assert isinstance(gate.jvm.RandomForest, JavaClass)

        # Make sure requesting a gateway again returns the same gateway
        gate2 = get_java_gateway()
        assert gate is gate2

        # Test getting a new gateway if needed
        gate3 = get_java_gateway(reuse=False)
        assert gate is not gate3

        # Make the server using the package version of lolo
        gate4 = get_java_gateway(reuse=False, skip_devel_version=True)
        java_import(gate4.jvm, "io.citrine.lolo.learners.*")
        assert isinstance(gate4.jvm.RandomForest, JavaClass)

    def test_memory(self, caplog, monkeypatch):
        # Set an environmental variable (local for this test)
        monkeypatch.setenv("LOLOPY_JVM_MEMORY", "4g")
        with caplog.at_level("DEBUG"):
                caplog.clear()  # Make sure logs are fresh
                get_java_gateway(reuse=False)  # Get a gateway
                # Make sure the memory amount appears in the logs
                assert any("Xmx4g" in msg for msg in caplog.messages)

    def test_find_lolo_jar(self, monkeypatch, tmp_path):
        lolopy_dir = tmp_path / "python" / "lolopy"
        installed_jar = lolopy_dir / "jar" / "lolo-jar-with-dependencies.jar"
        development_jar = tmp_path / "target" / "scala-2.13" / "lolo-jar-with-dependencies.jar"
        scala_dir = tmp_path / "src" / "main" / "scala" / "io" / "citrine" / "lolo"

        monkeypatch.setattr(loloserver, "_lolo_root", tmp_path)
        monkeypatch.setattr(loloserver, "_lolopy_dir", lolopy_dir)

        # Check resolution when the structure doesn't look like it has a development version
        with raises(RuntimeError):
            find_lolo_jar(skip_devel_version=False)

        with raises(RuntimeError):
            find_lolo_jar(skip_devel_version=True)

        installed_jar.parent.mkdir(parents=True)
        installed_jar.touch()
        assert find_lolo_jar(skip_devel_version=False) == installed_jar
        assert find_lolo_jar(skip_devel_version=True) == installed_jar

        # Emulate a scala file structure; now it sees a development version
        scala_dir.mkdir(parents=True)

        with raises(RuntimeError):  # Now looks for and can't find a development version
            find_lolo_jar(skip_devel_version=False)

        development_jar.parent.mkdir(parents=True)
        development_jar.touch()
        assert find_lolo_jar(skip_devel_version=False) == development_jar
        assert find_lolo_jar(skip_devel_version=True) == installed_jar
