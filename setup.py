"""Setup script for the package."""

# To use a consistent encoding
from codecs import open as copen
from os import path
import os
import re

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with copen(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def read(*parts):
    """Read file."""
    with copen(os.path.join(here, *parts), "r", encoding="utf8") as fp:
        return fp.read()


def find_version(*file_paths):
    """Find version of the package."""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("plot_keras_history", "__version__.py")

test_deps = [
    "pytest",
    "pytest-cov",
    "validate_version_code",
    "compress_json",
    "pytest_readme",
    "keras",
    "tensorflow",
    "extra_keras_metrics",
]

extras = {
    "test": test_deps,
}


setup(
    name="plot_keras_history",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description="A simple python package to print a keras NN training history.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/LucaCappelletti94/plot_keras_history",
    # Author details
    author="Luca Cappelletti",
    author_email="cappelletti.luca94@gmail.com",
    # Choose your license
    license="MIT",
    include_package_data=True,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
    ],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "matplotlib",
        "pandas",
        "scipy",
        "sanitize_ml_labels>=1.0.48",
    ],
    tests_require=test_deps,
    extras_require=extras,
)
