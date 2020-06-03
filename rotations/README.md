# Rotations

This package contains functions to rotate collections of 2D and 3D vectors.

Some of the functionality of this package is taken from the [Halotools](https://halotools.readthedocs.io/en/latest/) utilities sudmodule, reproduced here for convenience.    


## Description

This package contains tools to perform:

* rotations of 3D vectors,
* rotations of 2D vectors,
* and monte carlo rotations of 2D and 3D vectors.


## Requirements

In order to use the functions in this package, you will need the following python packages installed:

* numpy
* astropy


## Installation

Place this directory in your PYTHONPATH.  The various functions can then be imported as, e.g.:  

```
from rotations import rotate_vector_collection
```  

or for 2- and 3-D specific functions,

```
from rotations.rotations3d import rotation_matrices_from_vectors
``` 

You can run the testing suite for this package using the [pytest](https://docs.pytest.org/en/latest/) framework by executing the following command in the package directory:

```
pytest
```


contact:
duncanc@andrew.cmu.edu