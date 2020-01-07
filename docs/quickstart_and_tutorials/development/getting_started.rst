.. _getting_started_developers:

*************************
Contributing to Halotools
*************************

All halotools development happens in the github repository. To contribute, first clone the repo.
Then install the dependencies listed on the :ref:`step_by_step_install` page.


Code
====

Halotools contains a compiled component. To compile all cython (``.pyx``) files inplace run, ::

   python3 setup.py build_ext --inplace

If you modify a ``.pyx`` file use the same command to recompile it. Subsequent runs will only compile files whose source has changed and so will be much quicker.

Halotools also has comprehensive unit tests and uses pytest. To run all tests, assuming you are in the base of the repository, first change directory into ``halotools`` and then run ``pytest``. ::

   cd halotools
   pytest

If you have made a change and only want to run a subset of tests run,  ::

   pytest -k tests_matching_this_string_will_run

Run ``pytest --help`` for a full list of options.


Docs
====

First ensure that the halotools package and sphinx are installed. From the base of the repository run, ::

   pip3 install -e .
   pip3 install sphinx==1.3.1 # see docs/conf.py for the sphinx version

Then build documentation with, ::

   cd docs
   make html

You can see the built documentation in ``docs/_build/html/``. The easiest way to view it in your browser is to spin up a local server. One way to do this is to run, from the built directory, ::

   python3 -m http.server

The docs should then be viewable at ``localhost:8000`` (the port will be logged when you start the server).
