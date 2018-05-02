"""
Store/Access data in hdf5 file using some compression

Otherwise we have files which takes many gigabytes..

Saves hdf5 dataset in  'groupname/xxxx'
"""

import numpy as np
import logging
import h5py
import datetime
from settings import conf

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


def set_dataset(hdf, path, data):
    """Replace of set data in path of hdf file"""
    if path in hdf.keys():
        del hdf[path]
    dl = hdf.create_dataset(
        path, data=data, compression="gzip",
        compression_opts=7
    )
    return dl


def save(name, data, attrs={}):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "a") as data_file:
        path = f'{groupname}/{name}'
        ds = set_dataset(data_file, path, data)
        for k, v in attrs.items():
            ds.attrs[k] = v


def load_dataset(ds_name):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:
        return np.array(data_file[f'{groupname}/{ds_name}'])


def load_attrs(ds_name, keys):
    groupname = conf['groupname']
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "r") as data_file:
        ds = data_file[f'{groupname}/{ds_name}']
        attrs = {}
        for k in keys:
            attrs[k] = ds.attrs[k]
        return attrs


def load_timestamps(ds_name='timestamps'):
    timestamps = load_dataset(ds_name)

    time_x = []
    for t in timestamps:
        dt = datetime.date.fromtimestamp(t)
        time_x.append(dt)

    return time_x


def load_cru_grid():
    return load_dataset('grid')


def _printname(name):
    log.debug(name)


def print_paths():
    storage_name = conf['hdf5storage']
    with h5py.File(storage_name, "a") as data_file:
        data_file.visit(_printname)
