#Calculate correlation coefficient between CRU climatic variables

import logging
from load_datasets import load_data
import numpy

from numpy import corrcoef
from numpy import cov

def calculate_corr(datasets):

    tmp = datasets['tmp']
    pre = datasets['pre']
    vap = datasets['vap']
    pet = datasets['pet']

    X = numpy.stack((tmp, vap, pre, pet), axis=0)

    #print('cov')
    #print(cov(tmp, pre))
    print('corrcoef')
    print(corrcoef(X))
    

def main():
    # load hdf5 measurement data.
    timestamps, datasets = load_data()
    calculate_corr(datasets)

if __name__ =='__main__':
     main()