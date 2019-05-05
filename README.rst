Plot_keras_history
=============================

|travis| |coveralls| |sonar_quality| |sonar_maintainability| |code_climate_maintainability| |code_climate_test_coverage| |pip|

A simple python package to print a keras NN training history.

|example|

How do I get it?
----------------
Just type into your terminal:

.. code:: shell

   pip install plot_keras_history


Usage example
--------------

.. code:: python

    import matplotlib.pyplot as plt
    from plot_keras_history import plot_history
    import json

    history = json.load(open("tests/history.json", "r"))
    plot_history(history)
    plt.savefig('history.png')



.. |travis| image:: https://travis-ci.org/LucaCappelletti94/plot_keras_history.png
   :target: https://travis-ci.org/LucaCappelletti94/plot_keras_history

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/plot_keras_history/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/plot_keras_history

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=plot_keras_history.lucacappelletti&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/plot_keras_history.lucacappelletti

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=plot_keras_history.lucacappelletti&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/plot_keras_history.lucacappelletti

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/25fb7c6119e188dbd12c/maintainability
   :target: https://codeclimate.com/github/LucaCappelletti94/plot_keras_history/maintainability
   :alt: Maintainability

.. |code_climate_test_coverage| image:: https://api.codeclimate.com/v1/badges/25fb7c6119e188dbd12c/test_coverage
   :target: https://codeclimate.com/github/LucaCappelletti94/plot_keras_history/test_coverage
   :alt: Test Coverage

.. |pip| image:: https://badge.fury.io/py/plot_keras_history.svg
    :target: https://badge.fury.io/py/plot_keras_history

.. |example| image:: https://github.com/LucaCappelletti94/plot_keras_history/blob/master/history.png?raw=true