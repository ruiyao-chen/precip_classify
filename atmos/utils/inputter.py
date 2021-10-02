import scipy.io as sio
import torch
from torch import Tensor
import numpy as np
import h5py
from netCDF4 import Dataset as NetCDF_Dataset, num2date, date2num
import datetime


def load_matlab_data(dataPath: str, data='train', multi_label=False):
    assert data in ['train', 'test', 'valid']
    data_np = sio.loadmat(dataPath)
    # convert data into tensor and expand the dimension; torch.Size([82871, 3, 13])
    feature = data_np['sample']

    # if feature.ndim == 2:
    #     feature = np.expand_dims(feature, axis=1)

    # convert data into tensor and expand the dimension; torch.Size([82871, 1]), array([0, 1, 2, 3], dtype=uint8)
    label_v1 = data_np['label']
    # array([0, 1], dtype=uint8)
    label_v2 = data_np['label_v2']
    assert label_v1.shape[0] == label_v2.shape[0]
    if multi_label:
        if len(np.unique(label_v1)) == 2:
            label = np.concatenate([label_v1, label_v2], axis=1)
        else:
            label = np.concatenate([label_v2, label_v1], axis=1)
    else:
        label = label_v1

    return feature, label, label.shape


def load_hdf5_data(dataPath: str, data='train', bias=None, multi_label=False):
    with h5py.File(dataPath, 'r') as hf:
        feature = np.array(hf['inputs']).transpose()[:, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

        # if feature.ndim == 2:
        #     feature = np.expand_dims(feature, axis=1)[:, :, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

    if multi_label:
        label = np.ones((feature.shape[0], 2))
    else:
        label = np.ones((feature.shape[0], 1))

    return feature, label, label.shape


# def load_hdf5_data(dataPath: str, data='train', bias=None):
#     with h5py.File(dataPath, 'r') as hf, NetCDF_Dataset(bias, 'r', 'NETCDF3') as bias:
#         earthObs = np.array(hf['Data']['Earth_Obs_BT'])
#         bias_data = np.array(bias.variables['matrix']).transpose()
#         # bias_data = np.expand_dims(bias_data, axis=1)
#         bias_data = np.tile(bias_data, earthObs.shape[1]).reshape(earthObs.shape)
#         earthObs = earthObs - bias_data
#         sensorZenith = 1 / np.cos(np.deg2rad(np.array(hf['Geolocation']['SensorZenith'])*0.01))
#         b = earthObs.reshape((earthObs.shape[0], -1)).transpose()
#         c = sensorZenith.reshape((-1, 1))
#
#         feature = np.concatenate((b, c), axis=1)
#
#         if feature.ndim == 2:
#             feature = np.expand_dims(feature, axis=1)[:, :, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
#
#         label = np.ones(c.shape[0])
#     return feature, label, sensorZenith.shape


def save_hdf5_data(dataPath: str, data, name='Predict'):
    with h5py.File(dataPath, 'w') as hf:
        hf.create_dataset(name, data=data)


def load_netcdf_data(dataPath: str, device='cpu', data='train'):
    f = NetCDF_Dataset(dataPath)
    return f

def load_data(dataPath, data='train', bias=None, multi_label=False):
    if dataPath is None:
        return None, None, None

    data_type = dataPath.split(".")[-1]

    if data_type.lower() == 'mat':
        return load_matlab_data(dataPath, data, multi_label=multi_label)
    elif data_type.lower() == 'hdf':
        return load_hdf5_data(dataPath, data, bias, multi_label=multi_label)
    else:
        return load_matlab_data(dataPath, data, multi_label=multi_label)


def save_netcdf():
    # -----------------------
    nyears = 16
    unout = 'days since 2000-01-01 00:00:00'
    # -----------------------
    ny, nx = (250, 186)
    lon = np.linspace(9, 30, nx)
    lat = np.linspace(50, 60, ny)

    dataout = np.random.random((nyears, ny, nx))  # create some random data
    datesout = [datetime.datetime(2000 + iyear, 1, 1) for iyear in range(nyears)]  # create datevalues
    # =========================
    ncout = NetCDF_Dataset('myfile.nc', 'w', 'NETCDF3')  # using netCDF3 for output format
    ncout.createDimension('lon', nx)
    ncout.createDimension('lat', ny)
    ncout.createDimension('time', nyears)
    lonvar = ncout.createVariable('lon', 'float32', 'lon')
    lonvar[:] = lon
    latvar = ncout.createVariable('lat', 'float32', 'lat')
    latvar[:] = lat
    timevar = ncout.createVariable('time', 'float64', 'time')
    timevar.setncattr('units', unout)
    timevar[:] = date2num(datesout, unout)
    myvar = ncout.createVariable('myvar', 'float32', ('time', 'lat', 'lon'))
    myvar.setncattr('units', 'mm')
    myvar[:] = dataout
    ncout.close()
