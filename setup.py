from setuptools import setup, find_packages

from codecs import open
from os import path
import os

working_dir = path.abspath(path.dirname(__file__))
ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()
    
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="mli",
    version="0.0.1",
    description="Monotonic linear interpolation of neural nets.",
    install_requires=required,
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages("lib", exclude=["tests*"]),
    package_dir={"": "lib"},
)
