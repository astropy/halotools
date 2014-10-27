This is a Python wrapper of Peter Behroozi's `fast3tree` code, which was
(shamlessly) taken from his [Rockstar halo finder](https://bitbucket.org/gfcstanford/rockstar).

Usage
-----
On a 64-bit machine, just type 

    make

to make the c libraries. 

Here's an example of how you use it in your python code

    from fast3tree import fast3tree
    import numpy as np
    
    data = np.random.rand(10000, 3)
    with fast3tree(data) as tree:
        idx = tree.query_radius([0.5, 0.5, 0.5], 0.2)

    # do whatever you like with idx
