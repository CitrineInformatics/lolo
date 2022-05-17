from setuptools import setup
from glob import glob
import shutil
import os

# single source of truth for package version
version_ns = {}
with open(os.path.join("lolopy", "version.py")) as f:
    exec(f.read(), version_ns)
version = version_ns['__version__']

# Find the lolo jar
JAR_FILE = glob(os.path.join('..', 'target', 'scala-2.13', 'lolo-jar-with-dependencies.jar'))
if len(JAR_FILE) == 0:
    raise Exception('No Jar files found. Build lolo first by calling "make" or "cd ..; sbt assembly"')
elif len(JAR_FILE) > 1:
    raise Exception('Found >1 Jar file. Clean and rebuild lolopy: cd ..; sbt assembly')

# Copy the jar file to a directory at the same level as the package
jar_path = os.path.join('lolopy', 'jar')
if os.path.isdir(jar_path):
    shutil.rmtree(jar_path)
os.mkdir(jar_path)
shutil.copy(JAR_FILE[0], os.path.join(jar_path, 'lolo-jar-with-dependencies.jar'))

with open('README.md') as f:
    long_description = f.read()

# Make the installation
setup(
    name='lolopy',
    version=version,
    url='https://github.com/CitrineInformatics/lolo',
    maintainer='Max Hutchinson',
    maintainer_email='maxhutch@citrine.io',
    packages=[
        'lolopy',
        'lolopy.jar'  # Used for the PyPi packaging
    ],
    include_package_data=True,
    package_data={'lolopy.jar': ['*.jar']},
    install_requires=['scikit-learn', 'py4j'],
    description='Python wrapper for the Lolo machine learning library',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
