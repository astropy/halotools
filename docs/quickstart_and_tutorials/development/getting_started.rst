.. _getting_started:

*************************
Contributing to Halotools
*************************

All halotools development happens in the github repository. To contribute, first clone the repo.
Then install the dependencies listed on the *installation* page.

Halotools contains a compiled component. To compile all `.pyx` files inplace run

```
python3 setup.py build_ext --inplace
```

Halotools also has comprehensive unit tests using pytest. From the base of the repository first cd
into the source tree, then run `pytest`.

```
cd halotools
pytest
```

If you have made a change and only want to run a subset of tests run
```
pytest -k name_of_test_to_run
```
