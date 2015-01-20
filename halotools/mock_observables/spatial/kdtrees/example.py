from ckdtree import cKDTree
import numpy as np

def main():

    x = np.random.random((1000,3))
    y = np.random.random((10000,3))
    weights = np.random.random(len(x))
    
    r = np.arange(0,0.5,0.05)
    
    tree1 = cKDTree(x, leafsize=10)
    tree2 = cKDTree(y, leafsize=10)
    pairs = tree1.wcount_neighbors(tree2,r,sweights=weights)
    
if __name__ == '__main__':
    main()