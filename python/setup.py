from setuptools import setup
from pathlib import Path
from shutil import copy, rmtree

# single source of truth for package version
this_directory = Path(__file__).parent.absolute()
version_file = this_directory / "lolopy" / "__version__.py"
readme_file = this_directory / "README.md"

about = {}
if version_file.exists():
    exec(version_file.read_text(), about)
else:
    raise Exception(f"No version file found at {version_file}")
if readme_file.exists():
    about["long_description"] = readme_file.read_text()
else:
    raise Exception(f"No readme file found at {readme_file}")

# Find the lolo jar
JAR_FILE = list(this_directory.parent.glob("target/**/lolo-jar-with-dependencies.jar"))
if len(JAR_FILE) == 0:
    raise Exception(
        'No Jar files found. Build lolo first by calling "make" or "cd ..; sbt assembly"'
    )
elif len(JAR_FILE) > 1:
    raise Exception(
        'Found >1 Jar file. Clean and rebuild lolopy: "cd ..; sbt assembly"'
    )

# Copy the jar file to a directory at the same level as the package
jar_path = this_directory / "lolopy" / "jar"
if jar_path.exists():
    rmtree(jar_path)
jar_path.mkdir()
copy(JAR_FILE[0], jar_path / "lolo-jar-with-dependencies.jar")

# Make the installation
setup(
    name="lolopy",
    version=about["__version__"],
    python_requires=">=3.9",
    url="https://github.com/CitrineInformatics/lolo",
    maintainer="Maxwell Venetos",
    maintainer_email="mvenetos@citrine.io",
    packages=["lolopy", "lolopy.jar"],  # Used for the PyPi packaging
    include_package_data=True,
    package_data={"lolopy.jar": ["*.jar"]},
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=1.3.2,<1.7",
        "py4j>=0.10.9,<0.10.10",
    ],
    extras_require={
        "tests": ["pytest"],
    },
    description="Python wrapper for the Lolo machine learning library",
    long_description=about["long_description"],
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
