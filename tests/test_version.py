from plot_keras_history.__version__ import __version__
from validate_version_code import validate_version_code


def test_version():
    assert validate_version_code(__version__)