python setup.py build_ext -b

import cytools
import numpy as np

# x = np.random.random((100,800000))
x = np.ones((100,800000),dtype=np.float64)
y = np.arange(800000,dtype=np.float64)
z = cytools.cy_range_corrected(x,y)

%timeit cytools.cy_range_corrected(x,y)
zz=cytools.cy_range_corrected(x,y)

def range_corrected(  matrix, rang, step=1 ):
    rdim = matrix.shape[0]
    cdim = matrix.shape[1]
    result = np.zeros([rdim,cdim], dtype=np.float64)

    for i in range(0,cdim,step):  #step):
        result [:,i:i+step] = matrix[:,i:i+step] * rang[i]
        # result [:,i] = matrix[:,i] * rang[i]
    return result

%timeit range_corrected(x,y,1)

def mHz ( matrix,
            binWidth,
            shotNumber):

    const = 150
    rdim = matrix.shape[0]
    cdim = matrix.shape[1]
    result = np.zeros([rdim,cdim], dtype=np.float64)

    for i in range(rdim):
        for j in range(cdim):
            result[i,j] = matrix[i,j] * ( const /  binWidth[i] ) / shotNumber[i]
    return result

def brackground ( matrix,
            bkg):

    c = 0
    rdim = matrix.shape[0]
    cdim = matrix.shape[1]
    bkgdim = bkg.shape[1] - 1
    result =  np.zeros([rdim,cdim], dtype=np.float64)

    for i in range(rdim):
        c = 0
        for j in range(cdim):
            result[i,j] = matrix[i,j] - bkg[i,c]
            if c < bkgdim:
                c += 1
            else:
                c = 0
    return result


#########################
import cytools
import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

# vlim = {'analog-s':[0.2,16],'analog-p':[0.2,16],'analog':[0,0.7] }


# for date in pd.date_range('2018-08-01','2018-08-31',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
    # try:
date = pd.date_range('2018-08-28','2018-08-28',freq='d') #'2018-06-27','2018-07-14',freq='d'):
        #
binario =   Lidar(
    fechaI=date.strftime('%Y-%m-%d')[0],
    fechaF=date.strftime('%Y-%m-%d')[-1],
    scan='FixedPoint',
    output='raw'
)
# binario.read()

# binario.datos.loc(axis=1)[:,:,'photon-p'] * binario.datosInfo.loc[0,'BinWidth_photon-p']
# backup = [binario.datos.copy(), binario.datosInfo.copy()]
binario.datos        = backup[0]
binario.raw    = backup[0].copy()
binario.datosInfo   = backup[1]

binario.datos = binario.datos.resample('30s').mean()
binario.raw = binario.raw.resample('30s').mean()
binario.datosInfo = binario.datosInfo.resample('30s').mean()

#########################
print " Range Corrected cython Validation"

p = range_corrected(
        binario.datos.values,
        binario.datos.columns.get_level_values(0).values)
c = cytools.cy_range_corrected(
        binario.datos.values,
        binario.datos.columns.get_level_values(0).values)
print (p[np.isfinite(p)] == c[np.isfinite(c)]).all()

#########################
print "mHz cython Validation"
col = 'analog-p'

p = mHz(
        binario.datos.values,
        binario.datosInfo[ 'ADCBits_'+col ].values,
        binario.datosInfo[ 'ShotNumber_'+col ].values)
c = cytools.cy_mHz(
        binario.datos.values,
        binario.datosInfo[ 'ADCBits_'+col ].values,
        binario.datosInfo[ 'ShotNumber_'+col ].values)
print (p[np.isfinite(p)] == c[np.isfinite(c)]).all()
bol = ~(p[np.isfinite(p)] == c[np.isfinite(c)])
print p[np.isfinite(p)][bol][-50]
print c[np.isfinite(c)][bol][-50]
#########################
bkgd   = binario.datos.loc(axis=1) [
    binario.datos.columns.levels[0] [
        (binario.datos.columns.levels[0] > 18) &
        (binario.datos.columns.levels[0] < 21)
        ]
    ].groupby(level=(1,2), axis=1).mean()
bkgd [bkgd.isnull()] = 0

p = brackground(binario.datos.values, bkgd.values)
c = cytools.cy_brackground(binario.datos.values, bkgd.values)
print (p[np.isfinite(p)] == c[np.isfinite(c)]).all()

#########################



tmp = np.ones([4,10],dtype=np.float64)
input = np.ones(4,dtype=np.float64)*0.1
adcbits = np.ones(4,dtype=np.float64)*12
shot = np.ones(4,dtype=np.float64)*601

c = cytools.cy_mVolts( tmp, input, adcbits, shot)
