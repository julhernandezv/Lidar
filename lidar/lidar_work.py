# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('PDF')
import datetime as dt
import pandas as pd
import numpy as np
import struct
import sys, os, glob, locale
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.ticker import LogFormatterMathtext, LogLocator
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('font', family=fm.FontProperties(fname='/home/jhernandezv/Tools/AvenirLTStd-Book.ttf',).get_name(), size = 16)
plt.rc('font', family=fm.FontProperties(fname='/home/jhernandezv/Tools/AvenirLTStd-Book.ttf',).get_name(), size = 16)
typColor = '#%02x%02x%02x' % (115,115,115)
plt.rc('axes',labelcolor=typColor,edgecolor=typColor,)
plt.rc('axes.spines',right=False,top=False,left=True)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)


reload (sys)
sys.setdefaultencoding ("utf-8")
locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))

from pandas.plotting._core import MPLPlot

# ################################################################################
# ################################################################################
# Test Lectura Lidar

# filename = '/home/jhernandezv/Lidar/InfoLidar/AS0_180307-035517/RM1830703.553871'
# fileObj = open (filename, "rb")
#
# for i in range(8):
# 	print fileObj.readline()


#
#
# # # ################################################################################
# # # # ASCII
# files = glob.glob('InfoLidar/ASCII/20180310/*')
# ascii = Lidar(ascii=True,scan=False,output='raw')
# ascii.read(files,inplace=False)
# # #
# #
# # # ################################################################################
# # # instance
# files = glob.glob('InfoLidar/AS0_180307-035517/RM*')
# files = glob.glob('InfoLidar/ZS0_180223-131945/RM*')
import glob

# files = glob.glob('InfoLidar/ZS0_180709-151714/RM*')      # test
files = sorted(glob.glob('/home/jhernandezv/Lidar/InfoLidar/3Ds_180703-135028/RM*') )       # test2
# files = glob.glob('InfoLidar/3Ds_180704-105519/RM*')        # test3
from lidar.lidar import Lidar
instance = Lidar(scan='Zenith',output='raw')
df1,df2 = instance.read_folder(files)
t1,t2 = instance.read_file(files[0])
t3,t4 = instance.read_file(files[1])
t5,t6 = instance.read_file(files[-1],180)
# plt.close('all')
# instance.data.groupby(axis=1,level=1).median().boxplot(column=['analog-p','analog-s'])
# instance._save_fig(textSave='_boxplot_median_analog')
# plt.close('all')
# instance.data.groupby(axis=1,level=1).median().boxplot(column=['photon-p','photon-s'])
# instance._save_fig(textSave='_boxplot_median_photon')

# instance.data.quantile(np.arange(0,1,0.01)).groupby(axis=1,level=1).median()
a=6+5




# instance = Lidar(fechaI='2018-07-04',Fechaf='2018-07-04',scan='3D')
# ()

# backup = [instance.data, instance.dataInfo]
# instance.data        = backup[0]
# instance.raw    = backup[0]
# instance.dataInfo   = backup[1]
#
# instance.plot(textSave='_test5_',parameters=['photon-p'])
# instance.plot(textSave='_test5_log',parameters=['photon-p'],output='RCS',colorbarKind='Log')
# instance.plot(textSave='_test_4D',parameters=['photon-p'],output='RCS')
# # dd, di = instance.read_folder(files)
# # '-75.5686', '6.2680'io.data.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])

#
# instance.plot(textSave='_test_1',parameters=['photon-p'])
# instance.plot(textSave='_test_1',parameters=['photon-p'],output='RCS')
# instance.plot(textSave='_log_test_1',output='RCS',parameters=['photon-p'],colorbarKind='Log')
# instance.plot(textSave='_test_1',output='Ln(RCS)',parameters=['photon-p'])
# instance.plot(textSave='_test_1',output='dLn(RCS)',parameters=['photon-p'],colorbarKind='Anomaly')
# instance.plot(textSave='_test_1',output='fLn(RCS)',parameters=['photon-p'])
# instance.plot(textSave='_test_1',output='fdLn(RCS)',parameters=['photon-p'],colorbarKind='Anomaly')
# instance.plot(textSave='_test_1',output='dfLn(RCS)',parameters=['photon-p'],colorbarKind='Anomaly')
# instance.plot(textSave='_test_1',output='fdfLn(RCS)',parameters=['photon-p'],colorbarKind='Anomaly')

################################################################################
# FixedPoint#
################################################################################
# date = pd.date_range('2018-08-06','2018-08-06',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
# altura = 4.5
# instance = Lidar(fechaI=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='FixedPoint')
# instance.read()
# instance.data = instance.data.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])
# instance.raw = instance.data
# instance.dataInfo = instance.dataInfo.resample('30s').mean()
# instance.dataInfo = instance.dataInfo.reindex( pd.date_range(instance.data.columns.levels[0][0],instance.data.columns.levels[0][-1],freq='30s'))
# kwgs = dict( height=altura,)# background= bkg)
# instance.plot(**kwgs )

# instance.data.reindex( pd.date_range(instance.data.columns[0],instance.data.columns[-1],freq='30s'), axis=1)
# instance.data.reindex(pd.date_range(instance.data.columns.levels[0][0],instance.data.columns.levels[0][-1],freq='30s'))

#################################################################################
#Calculo CLA
################################################################################
import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))


from matplotlib import use
use('PDF')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatterMathtext, LogLocator

DATA_PATH = '/home/jhernandezv/Lidar/lidar/lidar/staticfiles/'

plt.rc(    'font',
    size = 20,
    family = FontProperties(
        fname = '{}/AvenirLTStd-Book.ttf'.format(DATA_PATH)
        ).get_name()
)

typColor = '#%02x%02x%02x' % (115,115,115)
plt.rc('axes',labelcolor=typColor, edgecolor=typColor,)#facecolor=typColor)
plt.rc('axes.spines',right=False, top=False, )#left=False, bottom=False)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)
plt.rc('figure.subplot', left=0, right=1, bottom=0, top=1)

#
# def cloud_filter(data,clouds):
#     tmpData = {}
#     for col in clouds.columns.levels[2]:
#         tmpData[col+'-p'] = data.xs(col+'-p',axis=1,level=2).mask(clouds.xs(col,axis=1,level=2) >=.95 )
#         tmpData[col+'-s'] = data.xs(col+'-s',axis=1,level=2).mask(clouds.xs(col,axis=1,level=2) >=.95 )
#     return pd.concat(tmpData,names=['Parameters','Dates']).unstack(0)


# vlim = {'analog-s':[0.2,16],'analog-p':[0.2,16],'analog':[0,0.7] }


# for date in pd.date_range('2018-06-01','2018-10-01',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
    # try:
# date = pd.date_range('2018-06-30','2018-06-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
date = pd.date_range('2018-10-18','2018-10-28',freq='d')
# date = pd.date_range('2018-10-01','2018-10-30',freq='d')
        #
instance =   Lidar(
    fechaI=date.strftime('%Y-%m-%d')[0],
    fechaF=date.strftime('%Y-%m-%d')[-1],
    scan='FixedPoint',
    output='raw'
)



instance.raw = instance.raw.resample('1T').mean()
instance.datosInfo = instance.datosInfo.resample('1T').mean()
# instance.raw = instance.raw.loc(axis=0)[
#         (instance.raw.index.hour>=6) & (instance.raw.index.hour<18)
#     ]
instance.get_output(output='RCS')


###############################
height=8
A=instance.datos.loc(axis=1)[:height,:,'analog-s']
A[A<.01] = .01
A[A>16] = 16

# Roll_min=A.rolling(window=10,center=True, closed='both',axis=0).mean()
Clouds=A.rolling(window=5,center=True, closed='both',axis=1).min()
Clouds=Clouds[Clouds>7]
Clouds=Clouds.sum(axis=1)
Clouds[Clouds>0]=1
# Se escoge la zona entre 200 y 400 metros y se le filtran los valores muy altos pues pueden ser nubes
Aerosols=instance.datos.loc(axis=1)[0.2:0.4,:,'analog-s']
Aerosols=Aerosols[Aerosols<=7]

# #plot:
# ax=Aerosols.mean(axis=1).plot(ylim=([0,1]),figsize=([10.5,3]),alpha=0.5)
# Aerosols.mean(axis=1).rolling(window=120, center=True, min_periods=10).mean(
    #).plot(ylim=([0,1]),figsize=([10.5,3]),ax=ax)
# pd.DataFrame(index=[Aerosols.index[0],Aerosols.index[-1]], data=[0.4,0.4]).plot(ax=ax, legend=False)
# Se promedia verticalmente la zona entre 200 y 400 y se hace una media movil de la serie obtenida
# Se considera que un umbral apropiado entre aerosol y no aerosol es 0.4
Aerosols=Aerosols.mean(axis=1).rolling(window=120, center=True, min_periods=10).mean()

Aerosols[Aerosols>=0.4]=1
Aerosols[Aerosols<0.4]=0

FilterDates = pd.concat({'Aerosols':Aerosols,'Clouds':Clouds},axis=1)

#Linealizando
# A = np.log10(A)
# Serie de RCS nocturno discriminada según el caso:
Case_AC=A[FilterDates['Clouds'] + FilterDates['Aerosols']==2] # Nubes y aerosoles
Case_aC=A[FilterDates['Clouds'] > FilterDates['Aerosols']]    # Nubes sin aerosoles
Case_Ac=A[FilterDates['Clouds'] < FilterDates['Aerosols']]    # Sin nubes con aerosoles
Case_ac=A[FilterDates['Clouds'] + FilterDates['Aerosols']==0] # Sin nubes ni aerosoles

rango   = instance.lidarProperties['RCS']['vlim']['analog-b']
bins    = np.logspace(np.log10(rango[0]),np.log10(rango[-1]),20)
# bins    = np.linspace(rango[0],rango[-1],20)
# bins    = np.log10(bins)
# np.histogram( bins=bins,range=range, density=True )


cases = {
    'Cloudy skies and\nhigh aerosol load':Case_AC,
    'Cloudy skies and\nlow aerosol load':Case_aC,
    'Cloud-free and\nhigh aerosol load':Case_Ac,
    'Cloud-free and\nlow aerosol load': Case_ac,
}
cases_name = ['Cloudy skies and\nlow aerosol load',
'Cloudy skies and\nhigh aerosol load',
 'Cloud-free and\nlow aerosol load',
 'Cloud-free and\nhigh aerosol load',
 ]

##########################################
#----------------LVD vs RCS--------------------
height=5
instance.get_output(output='RCS',totalSignal=True)
rcs=instance.datos.loc(axis=1)[.2:height,:,'analog-b']

instance.get_output(output='LVD')
lvd=instance.datos.loc(axis=1)[.2:height,:,'analog']
lvd = lvd.replace([np.inf, -np.inf], np.NaN)
# lvd[lvd>1] =1
# rcs[rcs>20] =20

rango_lvd   = [0.,1] #instance.lidarProperties['LVD']['vlim']['analog']
bins_lvd    = np.linspace(rango_lvd[0],rango_lvd[1],30)
rango_rcs   = [.01,20] #instance.lidarProperties['RCS']['vlim']['analog-b']
bins_rcs    = np.logspace(np.log10(rango_rcs[0]),np.log10(rango_rcs[-1]),30)

hist,xbn,ybn = np.histogram2d(
                x=lvd.values.reshape(lvd.values.size),
                y=rcs.values.reshape(rcs.values.size),
                bins=[bins_lvd,bins_rcs],
                range=[rango_lvd,rango_rcs],
                normed=False,
            )
hist = hist/np.float(hist.sum())*100.



# #########################################################
#----------Histograma 2d
plt.close('all')
fig = plt.figure(figsize=(6,6))
# Histograma por segmentos en la altura
plt.subplots_adjust(hspace=.3, )
ax={}
# height_discrete = np.arange(0.1,height,each)
# norm = Normalize(0,18)
# for c,key in enumerate(cases_name):
kwd = dict(
    # norm=Normalize(0,18),
    cmap='jet'
)
c=0
ax[c]       = fig.add_subplot(1, 1, c+1)

cf  = ax[c].pcolormesh(bins_lvd,bins_rcs,hist.T,**kwd)
# ax[c].set_title(key,loc='left')
ax[c].set_xlabel(r'LVD $(\delta^v)$', weight='bold')

ax[c].set_yscale('log')
# ax[c].set_xscale('log')
ax[c].set_ylabel(r'RCS $[mV*km^2]$',weight='bold') #, weight='bold')
ax[c].set_ylim(rango_rcs[0],rango_rcs[1])
ax[c].set_xlim(rango_lvd[0],rango_lvd[1])

cax      = fig.add_axes((1.02,.2,0.02,0.59))
cbar     = plt.colorbar(cf, cax = cax ,extend='max',**kwd)
cbar.set_label(r'Joint Probability $[\%]$',weight='bold')

instance._save_fig(localPath='Figuras/',textSave='Hist2d_LVDvsRCS',path='jhernandezv/Lidar/FixedPoint/Poster/Resultados/')

#########################################################
#----------Scatter
ht = np.array(
        list(
            lvd.columns.levels[0][
                (lvd.columns.levels[0]>.2) &
                (lvd.columns.levels[0]<height)]
        )* lvd.index.size
        )
ll = lvd.values.reshape(lvd.values.size)
rr = rcs.values.reshape(rcs.values.size)


plt.close('all')
fig = plt.figure(figsize=(6,6))
kwd = dict(
    norm=Normalize(0,height),
    cmap='jet',
    alpha=.3
)
c=0
ax[c]       = fig.add_subplot(1, 1, c+1)
cf = ax[c].scatter(ll,rr,c=ht,**kwd)
ax[c].set_yscale('log')
ax[c].set_ylabel(r'RCS $[mV*km^2]$',weight='bold') #, weight='bold')
ax[c].set_xlabel(r'LVD $(\delta^v)$', weight='bold')
ax[c].set_ylim(rango_rcs[0],rango_rcs[1])
ax[c].set_xlim(rango_lvd[0],rango_lvd[1])

cax      = fig.add_axes((1.02,.2,0.02,0.59))
cbar     = plt.colorbar(cf, cax = cax ,**kwd)
cbar.set_label(r'Height $[km]$',weight='bold')

instance._save_fig(localPath='Figuras/',textSave='Scatter_LVDvsRCS_Height',path='jhernandezv/Lidar/FixedPoint/Poster/Resultados/')



ax =lvd.stack(0).quantile(np.arange(0.01,1.,.01)).plot()
ax.set_ylabel(r'LVD $(\delta^v)$', weight='bold')
ax.set_xlabel(r'Percentile', weight='bold')
instance._save_fig(localPath='Figuras/',textSave='Percentile_LVD',path='jhernandezv/Lidar/FixedPoint/Poster/Resultados/')

ax=rcs.stack(0).quantile(np.arange(0.01,1.,.01)).plot()
ax.set_yscale('log')
ax.set_ylabel(r'RCS $[mV*km^2]$',weight='bold')
ax.set_xlabel(r'Percentile', weight='bold')
instance._save_fig(localPath='Figuras/',textSave='Percentile_RCS',path='jhernandezv/Lidar/FixedPoint/Poster/Resultados/')
#---------------------------------------------------
# Histograma

plt.close('all')
fig = plt.figure(figsize=(6,4))

ax       = fig.add_subplot(1, 1, 1)
for c,key in enumerate(cases_name):
    hist, bn    = np.histogram(
                    cases[key].values.reshape(cases[key].size),
                    bins=bins,
                    range=rango,
                    density=False
                )
    # print hist, bn
    hist = hist/np.float(hist.sum())*100.

    ax.plot(bins[:-1]+(bins[1]-bins[0])/2., hist, label=key,marker='o',alpha=.8)

ax.legend()
ax.set_ylabel(r'Relative Frequency $[\%]$',weight='bold')
ax.set_xlabel(r'RCS $[mV*km^2]$',weight='bold')
ax.set_xscale('log')
# plt.gca().xaxis.set_major_formatter(LogFormatterMathtext(10))
instance._save_fig(localPath='Figuras/',textSave='Hist_Cases',path='jhernandezv/Lidar/FixedPoint/RadiacionTest/')

#---------------------------------------------------
#bottom=0.,left=0.,right=1,top=1,wspace=.05,)
each=.1
Hist ={}

plt.close('all')
fig = plt.figure(figsize=(8,8))
# Histograma por segmentos en la altura
plt.subplots_adjust(hspace=.3, )
ax={}
height_discrete = np.arange(0.1,height,each)
norm = Normalize(0,18)
for c,key in enumerate(cases_name):
    ax[c]       = fig.add_subplot(2, 2, c+1)

    hist = np.empty((height_discrete.size,bins.size-1))
    for ix,h in enumerate(height_discrete):

        seccion         = cases[key].loc(axis=1)[h:h+each]
        hist[ix], bn    = np.histogram(
                            seccion.values.reshape(seccion.size),
                            bins=bins,
                            range=rango,
                            density=False
                        )
        hist[ix] = hist[ix]/np.float(hist[ix].sum())*100.
    Hist[key] =hist
    # print "shapes: \nbins:{}\nheight:{}\nhist:{}".format(bins.size,height_discrete.size,hist.shape)
    cf  = ax[c].pcolormesh(bins,np.arange(0.1,height+each,each),hist,norm=norm,cmap='jet')
    ax[c].set_title(key,loc='left')
    ax[c].set_xscale('log')
    if c in [0,2]:
        ax[c].set_ylabel(r'Height $[km]$', weight='bold')
    if c in [2,3]:
        ax[c].set_xlabel(r'RCS $[mV*km^2]$',weight='bold') #, weight='bold')
    ax[c].set_ylim(.1,5)

cax      = fig.add_axes((1.02,.2,0.02,0.59))
cbar     = plt.colorbar(cf, cax = cax , norm=norm,cmap='jet',extend='both')
cbar.set_label(r'Relative Frequency $[\%]$',weight='bold')

instance._save_fig(localPath='Figuras/',textSave='Hist_Height_Cases',path='jhernandezv/Lidar/FixedPoint/RadiacionTest/')

#---------------------------------------------------
# Histograma por segmentos en la altura
# Caso - Total
class MidpointNormalize(Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

each=.1
HistT ={}

plt.close('all')
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(hspace=.3, )
ax={}
height_discrete = np.arange(0.1,height,each)
norm = MidpointNormalize(midpoint=0.,
                            vmin=-10,
                            vmax=10)
for c,key in enumerate(cases_name):
    ax[c]       = fig.add_subplot(2, 2, c+1)

    histT   = np.empty((height_discrete.size,bins.size-1))
    hist    = np.empty((height_discrete.size,bins.size-1))
    for ix,h in enumerate(height_discrete):
        seccion         = A.loc(axis=1)[h:h+each]
        histT[ix],bn    = np.histogram(
                            seccion.values.reshape(seccion.size),
                            bins=bins,
                            range=rango,
                            density=False
                        )
        histT[ix] = histT[ix]/np.float(histT[ix].sum())*100.
        seccion         = cases[key].loc(axis=1)[h:h+each]
        hist[ix], bn    = np.histogram(
                            seccion.values.reshape(seccion.size),
                            bins=bins,
                            range=rango,
                            density=False
                        )
        hist[ix] = hist[ix]/np.float(hist[ix].sum())*100.
    hist = hist - histT
    HistT[key] =hist
    # print "shapes: \nbins:{}\nheight:{}\nhist:{}".format(bins.size,height_discrete.size,hist.shape)
    cf  = ax[c].pcolormesh(bins,np.arange(0.1,height+each,each),hist,norm=norm,cmap='seismic')
    ax[c].set_title(key,loc='left')
    ax[c].set_xscale('log')
    if c in [0,2]:
        ax[c].set_ylabel(r'Height $[km]$', weight='bold')
    if c in [2,3]:
        ax[c].set_xlabel(r'RCS $[mV*km^2]$',weight='bold') #, weight='bold')
    ax[c].set_ylim(.1,5)

cax      = fig.add_axes((1.02,.2,0.02,0.59))
cbar     = plt.colorbar(cf, cax = cax , norm=norm,cmap='seismic')
cbar.set_label(r'Relative Frequency Anomaly $[\%]$',weight='bold')

instance._save_fig(localPath='Figuras/',textSave='Hist_Height_Cases_Diff',path='jhernandezv/Lidar/FixedPoint/RadiacionTest/')










#########################################################
# Radiación

import glob as gb
def lee_Pira(fechaInicio, fechaFin, Piranometro='siata'):
    path    = "torresiata@192.168.1.62:/mnt/ALMACENAMIENTO/piranometros/{}/".format(Piranometro)
    fechas  = pd.date_range(fechaInicio,fechaFin,freq='D')
    data    = []
    for fecha in fechas:
        archivo = fecha.strftime("LOG%y%m%d*.csv")
        print archivo
        os.system("scp {}{} Datos/".format(path,archivo))
        os.system("scp {}{} Datos/".format(path,archivo))
        archivos = gb.glob('Datos/'+archivo)
        for arc in archivos:
            data.append(
                pd.read_csv(arc,
                        sep=';',
                        usecols=[1,2,5],
                        skiprows=4,
                        parse_dates=[[0,1]],
                        names=['Fecha','Hora','Radiacion']
                ).set_index('Fecha_Hora')
            )
            # os.system('rm {}'.format(arc))
    data = pd.concat(data)
    data.mask(data<0,inplace=True)
    if Piranometro=='siata':
        data.index = data.index - dt.timedelta(hours=5)
    return data

# FechaiInicio='20181018 00:00'
# FechaFinal='20181028 23:59'
pira=lee_Pira(date[0], date[-1])
pira = pira.groupby(pira.index.hour).apply(lambda x: x -x.mean())

plt.close('all')
fig = plt.figure(figsize=(9,9))
for c,key in enumerate(cases.keys()):
    ax[c]      = fig.add_subplot(2, 2, c+1)
    idx = cases[key].index
    ax[c].scatter(RCS400mean.loc[idx].values,pira.loc[idx].values,alpha=.5)# label=key,
    ax[c].set_title(key,loc='left')
    ax[c].set_xscale('log')
    ax[c].set_xlim(rango[0],rango[-1])
    if c in [0,2]:
        ax[c].set_ylabel(r'Radiacion $[W/m^2]$',weight='bold')
    if c in [2,3]:
        ax[c].set_xlabel('RCS promedio 0.2-0.4 km')

# ax.legend()

# plt.gca().xaxis.set_major_formatter(LogFormatterMathtext(10))
instance._save_fig(localPath='Figuras/',textSave='Scatter_Cases2',path='jhernandezv/Lidar/FixedPoint/RadiacionTest/')


#---------------------------------------------
# Mapa

import AirNew as Air

# est = pd.read_csv('EstacionesBogota.csv')
reload(Air)
self= Air.Air()
#~ ext = ['latmax','latmin','lonmin','longmax']
# 75.4W a 75.7W
# 6.1N a 6.4N
# ext=[6.515,5.975,-75.725,-75.1255]
ext=[6.4,6.1,-75.7,-75.401]
scatter = [
	[-75.6443, -75.5742, -75.5887],
	[6.1681, 6.2422, 6.2593]
]
station = ['I.E. Concejo de Itagüí', 'AMVA', 'Torre Siata']
self.Plot_Mapa2(
	textsave='_Lidar',
	path='jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/FixedPoint/Poster/',
	hillshade=True,
	clim=[1300,3400],
	macroLocalizacion=False,
	georef=ext,
	extendSquare=False,
	saveFig=False
)

self.X,self.Y		= self.m(scatter[0] ,scatter[1])
self.m.scatter(self.X,self.Y,s=700,facecolor='g',edgecolor=(0.0078,0.227,0.26),zorder=10)

self.X,self.Y		= self.m(-75.578526, 6.201328)
self.m.scatter(self.X,self.Y,s=700,facecolor='b',edgecolor=(0.0078,0.227,0.26),zorder=11)

self.X,self.Y		= self.m( -75.5887, 6.2593)
self.m.scatter(self.X,self.Y,s=700,facecolor='k',marker='x',edgecolor=(0.0078,0.227,0.26),zorder=12)

leg={}
leg['Ceilometers'] = plt.Line2D( #\nPyranometers
		(0,1),(0,0),
		ls='',
		marker='o',
		markersize=18,
		mfc='g',
		mec=(0.0078,0.227,0.26),
		lw=2,
		fillstyle='full'
	)
leg['Scanning Lidar'] = plt.Line2D(
		(0,1),(0,0),
		ls='',
		marker='o',
		markersize=18,
		mfc='b',
		mec=(0.0078,0.227,0.26),
		lw=2,
		fillstyle='full'
	)
leg['Radiometer'] = plt.Line2D(
		(0,1),(0,0),
		ls='',
		marker='x',
		markersize=18,
		mfc='k',
		mec=(0.0078,0.227,0.26),
		lw=2,
		fillstyle='full'
	)
legend =self.ax[0].legend(
	leg.values(),
	leg.keys(),
	bbox_to_anchor=(1,.15),
	fontsize=22,
	loc='lower right',
	title='Remote Sensors',
	labelspacing=.8)
# self.leg.get_frame().set_edgecolor('w')
# for text in self.leg.get_texts(): plt.setp(text, color = (0.45, 0.45, 0.45))
legend.get_title().set_fontsize(22)
# plt.setp(self.leg.get_title(),fontsize=self.fontsize,weight='bold',color=(0.45, 0.45, 0.45))

textsave='_Lidar'
path='jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/FixedPoint/Poster/'
plt.savefig('Figuras/Mapa%s.pdf' %textsave, bbox_inches='tight')
os.system('scp Figuras/Mapa%s.pdf %s' %(textsave,path))



###',################33
instance.get_output(output='LVD')

lvd = instance.datos.loc(axis=1) [
    instance.datos.columns.levels[0] [
        instance.datos.columns.levels[0] < height
    ]
]

instance.get_output(output='RCS')



instance.datos = instance.datos.loc(axis=1) [
    instance.datos.columns.levels[0] [
        instance.datos.columns.levels[0] < height
    ]
]

# Suavizado
#Espacial
instance.datos = instance.datos.groupby(
                    level=[1,2], axis=1
                    ).rolling(16,
                        center=True,
                        min_periods=1,
                        axis=1
                        ).mean()
instance.datos.columns = instance.datos.columns.droplevel([0,1])
# Temporal
instance.datos = instance.datos.rolling(6,
                        center=True,
                        min_periods=1).mean()


instance.datos = instance.datos.resample('10T').mean()

lvd = lvd.resample('10T').mean()
backup = instance.datos.copy()
#------------------------
# Varianza Maxima
# ------------------------

# instance.datos = cloud_filter(backup, lvd)

instance.datos = instance.datos.groupby(
                    level=[1,2], axis=1
                    ).rolling(54,
                        center=True,
                        min_periods=1,
                        axis=1).var()
instance.datos.columns = instance.datos.columns.droplevel([0,1])

instance.plot(
    df=instance.datos,
    path='claTest',
    textSave='_VM',
    colorbarKind='Linear',
    height=height,
    colormap='jet',
    cbarLabel=r'$\sigma(RCS)$')

vm = instance.datos.groupby(level=[1,2],axis=1).idxmin(axis=1)
for col in vm.columns:
    vm[col] = vm[col].str[0]
#------------------------
# Gradiente minimo
#------------------------
# instance.datos = cloud_filter(backup, lvd)
instance.datos = backup.copy()
dr  = (instance.datos.columns.levels[0][2] - instance.datos.columns.levels[0][0])
instance.datos = instance.datos.groupby(level=[1,2],axis=1).diff(axis=1,periods=2) /dr


# instance.plot(
#     df=instance.datos,
#     path='claTest',
#     textSave='_GM',
#     colorbarKind='Anomaly',
#     height=height,
#     colormap='seismic',
#     cbarLabel=r'$\delta(RCS)$',
#     # vlim=[-3,15]
#     )

#get cla
gm = instance.datos.groupby(level=[1,2],axis=1).idxmin(axis=1)
for col in gm.columns:
    gm[col] = gm[col].str[0]


#################################################################################
#Calculo CLA
################################################################################
# ps -fea | grep {}


from Funciones_Lectura import lee_Ceil,lee_data_ceil
# import pandas as pd
import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

# vlim = {'analog-s':[0.2,16],'analog-p':[0.2,16],'analog':[0,0.7] }


for date in pd.date_range('2018-10-10','2018-11-30',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
# for date in pd.date_range('2018-10-07','2018-10-10',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
    try:
# date = pd.date_range('2018-10-07','2018-10-10',freq='d')
# date = pd.date_range('2018-10-23','2018-10-28',freq='d')
# date = pd.date_range('2018-06-30','2018-06-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):


        # now = dt.datetime.now()
        instance =   Lidar(
                fechaI=date.strftime('%Y-%m-%d'),
                fechaF=date.strftime('%Y-%m-%d'),
                # fechaI=date[0].strftime('%Y-%m-%d'),
                # fechaF=date[-1].strftime('%Y-%m-%d'),
                # scan='3D',
                scan='FixedPoint',
                output='raw',
                # user='torresiata',
                # path='CalidadAire/Lidar/'
                # source='miel',
            )
        #
        # instance.datos = instance.datos.resample('30s').mean()
        # instance.raw = instance.raw.resample('30s').mean()
        # instance.datosInfo = instance.datosInfo.resample('30s').mean()
        #


        kwgs = dict(
            height=14,
            # height=12,
            # path= 'Poster/RCS/',
            # path= date.strftime('%m-%d'),
            cla=False, #True
            user='jhernandezv',
        )

    # for date in pd.date_range('2018-10-07','2018-10-10',freq='d'):
        # instance.get_output(output='RCS',totalSignal=True)
        instance.plot(
            output='RCS',
            dates=instance.datos[date.strftime('%Y-%m-%d')].index,
            path= 'Poster/RCS/',
            # parameters=['analog-s','analog-p','analog-b'],
            **kwgs
        )
        instance.plot(
            output='LVD',
            dates=instance.datos[date.strftime('%Y-%m-%d')].index,
            path= 'Poster/LVD/',
            # parameters=['analog-s','analog-p','analog-b'],
            **kwgs
        )
        del instance
# instance.plot(
#     output='P(r)',
#     totalSignal=True,
#
#     **kwgs
# )
# instance.plot(
#     output='S(r)',
#     totalSignal=True,
#     # parameters=['analog-s','analog-p','analog-b'],
#     **kwgs
# )

    except:
        import traceback
        traceback.print_exc()
        pass





instance.plot(output = 'raw',**kwgs )


instance.plot(output = 'S(r)',totalSignal=True,**kwgs )

instance.plot(
    output='RCS',
    **kwgs
)
instance.plot(
    output='LVD',
    **kwgs
)



import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

# vlim = {'analog-s':[0.2,16],'analog-p':[0.2,16],'analog':[0,0.7] }


# for date in pd.date_range('2018-06-01','2018-10-11',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
#     try:
date = pd.date_range('2018-06-29','2018-06-29',freq='d')
# date = pd.date_range('2018-10-07','2018-10-08',freq='d')
# '10-07'
# '11-12 06:00' '20'
# date = pd.date_range('2018-06-30','2018-06-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
        #
instance =   Lidar(
    fechaI=date[0].strftime('%Y-%m-%d'),
    fechaF=date[-1].strftime('%Y-%m-%d'),
    # scan='FixedPoint',
    scan='3D',
    output='raw',
    user='torresiata',
    source='miel'
)


# instance.get_output(output='RCS')
# instance.datos.loc(axis=1)[:,:,'photon-p'] * instance.datosInfo.loc[0,'BinWidth_photon-p']
# backup = [instance.datos.copy(), instance.datosInfo.copy()]
# # instance.datos        = backup[0]
# # instance.raw    = backup[0].copy()
# # instance.datosInfo   = backup[1]'cython_test', #
#
# instance.datos = instance.datos.resample('30s').mean()
# instance.raw = instance.raw.resample('30s').mean()
# instance.datosInfo = instance.datosInfo.resample('30s').mean()


# '2018-07-29 09:58'
kwgs = dict(
    height=1.5,
    # height=12,
    # path= 'Poster/LVD/',
    path= 'Poster/Scanning/',
    # path= date.strftime('%m-%d'),
    # dates=instance.datos['2018-06-29 09:58'].index,
    textSave='_1.5km',
    cla=False,
    user='jhernandezv',
    saveFig=False,
)
instance.plot(
    output = 'RCS',
    **kwgs )
xx,yy,zz = instance.X, instance.Y, instance.Z
dx = xx[0,1]-xx[0,0]
xx += dx
xx = xx[:-1,:-1]
dy = xx[1,0]-xx[0,0]
yy += dy
yy = yy[:-1,:-1]

kwgs['path'] ='jhernandezv/Lidar/FixedPoint/Poster/Scannings'
instance.axes[0].contour(xx,yy, zz, [0.35], colors='k',)
instance._save_fig(**kwgs)

instance.plot(
    output = 'LVD',
    **kwgs )

instance.axes[0].contour(xx,yy, zz, [0.35], colors='k',)
instance._save_fig(**kwgs)


kwgs['textSave'] = '_09-00'
kwgs['height'] = 7
instance.plot(
    output = 'RCS',
    **kwgs )
instance.plot(
    output = 'LVD',
    **kwgs )

# instance.plot(output = 'raw',**kwgs )
# instance.plot(output = 'S(r)',totalSignal=True,**kwgs )

# instance.plot(output = 'P(r)',
#     totalSignal=True,
#     vlim=[0,135],
#     parameters=['photon-s','photon-p'],
#     **kwgs )
#
# instance.plot(output = 'P(r)',
#     totalSignal=True,
#     vlim=[5,34],
#     parameters=['analog-s','analog-p'],
#     **kwgs )
#
# instance.plot(output = 'LVD',
#     totalSignal=True,
#     vlim=[0.25,1],
#     **kwgs )

instance.plot(
    output='RCS',
    parameters=['analog-b'],
    vlim = [.15,20],#[1,20],
    **kwgs
)

instance.plot(
    parameters=['analog-s','analog-p'],
    vlim=[.1,16], #[1,16]
    output='RCS',
    **kwgs
)

instance.plot(
    parameters=['photon-s','photon-p'],
    vlim=[9,200], #[15,350]
    output='RCS',
    **kwgs
)

instance.plot(
    output='RCS',
    parameters=['photon-b'],
    vlim = [10,200],
    **kwgs
)
import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

# vlim = {'analog-s':[0.2,16],'analog-p':[0.2,16],'analog':[0,0.7] }


# for date in pd.date_range('2018-06-01','2018-10-11',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
#     try:
# date = pd.date_range('2018-08-17','2018-08-17',freq='d')[0]
#Dark Measurement
date = pd.date_range('2018-11-29','2018-11-29',freq='d') #'2018-06-27','2018-07-14',freq='d'):
        #
bkg =   Lidar(
    fechaI=date[0].strftime('%Y-%m-%d'),
    fechaF=date[-1].strftime('%Y-%m-%d'),
    scan='FixedPoint',
    # scan='3D',
    output='P(r)'
)

#
date = pd.date_range('2018-11-17','2018-11-17',freq='d') #'2018-06-27','2018-07-14',freq='d'):
        #
instance =   Lidar(
    fechaI=date[0].strftime('%Y-%m-%d'),
    fechaF=date[-1].strftime('%Y-%m-%d'),
    scan='FixedPoint',
    # scan='3D',
    output='raw'
)



instance.datos = instance.datos.resample('30s').mean()
instance.raw = instance.raw.resample('30s').mean()
instance.datosInfo = instance.datosInfo.resample('30s').mean()


instance.get_output(output='P(r)')
instance.datos = instance.datos - bkg.datos.mean().loc[instance.datos.columns.levels[0]]
instance.background
instance.RCS
instance.datos[instance.datos<=0] = .01

kwgs = dict(
    height=20,
    # height=12,
    path= 'bkgtest/blackM-far',
    # path= date.strftime('%m-%d'),
    # cla=True
    # operational=True,
    parameters=['analog-s','analog-p'],
    textSave='-dark-bkgFar_Max_corrected',
    colorbarKind='Log',
)
instance.output='RCS'
instance.plot(
    df=instance.datos,
    **kwgs
)


##################################################

from matplotlib import use
use('PDF')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatterMathtext, LogLocator

DATA_PATH = '/home/jhernandezv/Lidar/lidar/lidar/staticfiles/'

plt.rc(    'font',
    size = 20,
    family = FontProperties(
        fname = '{}/AvenirLTStd-Book.ttf'.format(DATA_PATH)
        ).get_name()
)

typColor = '#%02x%02x%02x' % (115,115,115)
plt.rc('axes',labelcolor=typColor, edgecolor=typColor,)#facecolor=typColor)
plt.rc('axes.spines',right=False, top=False, )#left=False, bottom=False)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)
plt.rc('figure.subplot', left=0, right=1, bottom=0, top=1)

var='analog-s'
date='2018-10-07 09:30'

def plot_X(var,date):
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    ax       = fig.add_subplot(1, 1, 1)
    x = instance.datos[date].loc(axis=1)[:,:,var]
    ax.plot(x.columns.levels[0].values,x.values[0,:],label='P(r)',color='k',alpha=.8)

    x1 = bkg.datos.loc(axis=1)[:,:,var]
    ax.plot(x1.columns.levels[0].values,x1.values[0,:],label='Dark Measurement 1',color='b',ls='--',alpha=.8)
    ax.plot(x1.columns.levels[0].values,x1.values[1,:],label='Dark Measurement 2',color='c',ls='-.',alpha=.8)

    ax.axhline(x.values[0,:5].mean(),label='bkg-Near',color='r',alpha=.8)
    ax.axhline( x.loc(axis=1)[16:18,:,:].values[0,:].mean(),label='bkg-Far',color='g',alpha=.8)
    ax.axhline( x.loc(axis=1)[16:18,:,:].values[0,:].max(),label='bkg-Far_max',color='orange',alpha=.8)
    ax.axhline( x.loc(axis=1)[16:18,:,:].values[0,:].min(),label='bkg-Far_min',color='y',alpha=.8)
    ax.legend()
    ax.set_xlabel(r'Range $[km]$',weight='bold')
    ax.set_ylabel(r'$[mV]$',weight='bold') #r'RCS $[mV*km^2]$'
    ax.set_xlim(0,20)
    ax.set_ylim(4.75,6.25)
    ax.set_title('%s_%s' %(date,var))

    # ax.set_xscale('log')
    # plt.gca().xaxis.set_major_formatter(LogFormatterMathtext(10))
    instance._save_fig(localPath='Figuras/',textSave='Perfil_%s_%s' %(date,var),path='jhernandezv/Lidar/FixedPoint/bkgtest/')


plot_X(var='analog-s', date='2018-10-07 09:30')
plot_X(var='analog-s', date='2018-10-07 19:30')
plot_X(var='analog-p', date='2018-10-07 09:30')
plot_X(var='analog-p', date='2018-10-07 19:30')
# instance.get_output(output='RCS')
# instance.datos.loc(axis=1)[:,:,'photon-p'] * instance.datosInfo.loc[0,'BinWidth_photon-p']
# backup = [instance.datos.copy(), instance.datosInfo.copy()]
# # instance.datos        = backup[0]
# # instance.raw    = backup[0].copy()
# # instance.datosInfo   = backup[1]'cython_test', #
#

#

# instance.get_output('P(r)')

kwgs = dict(
    height=.05,
    # height=12,
    path= 'bkgtest/',
    # path= date.strftime('%m-%d'),
    cla=False, #True
    # user='jhernandezv',
    # textSave='_Black_Measurement',
)

instance.plot(output = 'P(r)',
    # totalSignal=True,
    # vlim=[20,70], #[0,135],
    parameters=['photon-s','photon-p'],
    **kwgs )
#
instance.plot(output = 'P(r)',
    # totalSignal=True,
    # vlim=[5,7], #[5,34],
    parameters=['analog-s','analog-p'],
    **kwgs )

kwgs = dict(
    height=4.5,
    # height=12,
    path= 'bkgtest',
    # path= date.strftime('%m-%d'),
    # cla=True
    # operational=True,
    textSave='_Identifier-2',
)
instance.plot(
    output='S(r)',
    **kwgs
)
instance.plot(
    output='RCS',
    **kwgs
)
instance.plot(
    output='LVD',
    **kwgs
)
# instance.plot(output = 'raw',**kwgs )
# instance.plot(output = 'S(r)',totalSignal=True,**kwgs )

# instance.plot(output = 'P(r)',
#     totalSignal=True,
#     vlim=[0,135],
#     parameters=['photon-s','photon-p'],
#     **kwgs )
#
# instance.plot(output = 'P(r)',
#     totalSignal=True,
#     vlim=[5,34],
#     parameters=['analog-s','analog-p'],
#     **kwgs )
#
# instance.plot(output = 'LVD',
#     totalSignal=True,
#     vlim=[0.25,1],
#     **kwgs )

instance.plot(
    output='RCS',
    parameters=['analog-b'],
    vlim = [.15,20],#[1,20],
    **kwgs
)

instance.plot(
    parameters=['analog-s','analog-p'],
    vlim=[.1,16], #[1,16]
    output='RCS',
    **kwgs
)

instance.plot(
    parameters=['photon-s','photon-p'],
    vlim=[9,200], #[15,350]
    output='RCS',
    **kwgs
)

instance.plot(
    output='RCS',
    parameters=['photon-b'],
    vlim = [10,200],
    **kwgs
)

        instance.plot( output='Ln(RCS)', **kwgs )

        instance.plot(output='dLn(RCS)',  **kwgs)

        instance.plot(output='fLn(RCS)', **kwgs)

        instance.plot(output='fdLn(RCS)',  **kwgs)

        instance.plot(output='dfLn(RCS)', **kwgs)

        instance.plot(output='fdfLn(RCS)' , **kwgs)

        for location in ['amva','siata','itagui']:
            # instance = Lidar(fechaI='20180801',Fechaf='20180801',scan='FixedPoint',output='raw')
            ceil = lee_Ceil(
                ceilometro=location,
                Fecha_Inicio=instance.datosInfo.index[0].strftime('%Y%m%d %H:%M'), Fecha_Fin=instance.datosInfo.index[-1].strftime('%Y%m%d %H:%M') ) #'%Y-%m-%d %H:%M:%S'
            # ceil         = lee_Ceil(ceilometro=location,  Fecha_Inicio='20180801 10:45', Fecha_Fin='20180801 13:25' )
            ceil.columns            = (ceil.columns+1)/100.
            ceil[ceil < 100]        = 100
            ceil[ceil.isnull()]     = 100
            if ceil.size > 0:
                instance.plot_lidar(
                    ceil.index,ceil.columns,ceil.T,
                    textSave='Ceilometro_'+location,
                    colorbarKind='Log',
                    vlim=[10,13000],
                    cbarlabel=r'Attenuated Backscatter $[10^{-9}m^{-1}sr^{-1}]$',
                    path='jhernandezv/Lidar/FixedPoint/' +kwgs['path'],
                    colormap = instance.ceilCmap
                )
    except:
        import traceback
        traceback.print_exc()
        pass
# #

# instance.plot( output='Ln(RCS)', **kwgs )
################################################################################
# Ceilomoetro
# from Funciones_Lectura import lee_Ceil,lee_data_ceil
# # #
# fecha_test = pd.date_range('2018-08-03 09:35:00', '2018-08-03 18:56:30',freq='30s')#instance.data.columns.levels[0]
# ceil =lee_data_ceil('amva',fecha_test[0].strftime('%Y-%m-%d %H:%M:%S'),fecha_test[-1].strftime('%Y-%m-%d %H:%M:%S') ) #fecha_test[0].strftime('%Y%m%d %H:%M'),fecha_test[-1].strftime('%Y%m%d %H:%M'),'amva')
# ceil.columns = (ceil.columns+1)/100.
# ceil[ceil < 100] = 100
# ceil[ceil.isnull()] = 100
# #
# instance.plot_lidar(ceil.index,ceil.columns,ceil.T,textSave='Ceilometro_AMVA',colorbarKind='Log',vlim=[10,13000],cbarlabel='Intensidad Backscatter $[10^{-9}m^{-1}sr^{-1}]$',**kwgs)
##############################################################################
        #

        # instance.plot(output='dLn(RCS)',colorbarKind='Anomaly',  **kwgs)

        # instance.plot(output='fLn(RCS)', **kwgs)

        # instance.plot(output='fdLn(RCS)',colorbarKind='Anomaly',  **kwgs)

        # instance.plot(output='dfLn(RCS)', colorbarKind='Anomaly', **kwgs)

        # instance.plot(output='fdfLn(RCS)', colorbarKind='Anomaly',  **kwgs)

        #10 km
        # kwgs = dict(  height=10,textSave='_10km',path='10km')# background= bkg)
        #
        # instance.plot(output = 'P(r)', **kwgs )
        #
        # instance.plot( output='RCS', **kwgs )
        #
        #
        # instance.plot( output='Ln(RCS)', **kwgs )
        # # #
        #
        # # instance.plot(output='dLn(RCS)',colorbarKind='Anomaly',  **kwgs)
        #
        # instance.plot(output='fLn(RCS)', **kwgs)
        #
        # # instance.plot(output='fdLn(RCS)',colorbarKind='Anomaly',  **kwgs)
        #
        # # instance.plot(output='dfLn(RCS)', colorbarKind='Anomaly', **kwgs)
        #
        # instance.plot(output='fdfLn(RCS)', colorbarKind='Anomaly',  **kwgs)
        #
        # kwgs.pop('textSave')
        # instance.plot(textSave='_log_10km', output='RCS',colorbarKind='Log',  **kwgs)


    # except:
    #     pass


# # ################################################################################
# backgroud = '2018-06-30 19:07'
# bkg = pd.read_csv('Background_test.csv',index_col=0,header=[0,1])
# bkg = pd.read_csv('Background.csv',index_col=0,header=[0,1])
# bkg.columns.set_levels(map(lambda x: int(x),bkg.columns.levels[0].values), level=0,inplace=True)
# # bkg = bkg.rolling(30,center=True,min_periods=1).mean()
# bkg = {'analog-p':5.307, 'analog-s':4.9846, 'photon-p':0.26578, 'photon-s':0.26578}

# x = bkg.loc[bkg.index>15,pd.IndexSlice[:,'photon-s']]
# x[x>0].min() - x[x>0].min().min() *18**2

# bkg.loc[:,pd.IndexSlice[:,'analog-s']]  = 4.9846
# bkg.loc[:,pd.IndexSlice[:,'analog-p']]  = 5.307
# bkg.loc[:,pd.IndexSlice[:,['photon-p','photon-s']] ]  = 0.26578#0.13289

# bkg.loc[:,pd.IndexSlice[:,'analog-p']]  =
# ################################################################################
# from Funciones_Lectura import lee_Ceil,lee_data_ceil
# import pandas as pd
import pandas as pd
import numpy as np
import datetime as dt
import os, locale
# os.system('rm lidar/lidar/*.pyc')
from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))


# for date in pd.date_range('2018-02-23','2018-10-11',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
#     try:
date = pd.date_range('2018-10-11','2018-10-11',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):

instance = Lidar(
fechaI=date.strftime('%Y-%m-%d'),
fechaF=date.strftime('%Y-%m-%d'),
scan='Azimuth',
output='raw'
)

# # backup = [instance.raw, instance.dataInfo]
# # instance.data        = backup[0]
# # instance.raw    = backup[0]
# # instance.dataInfo   = backup[1]
#
#
# kwgs = dict(parameters=['photon-p'], dates=instance.dataInfo.index, make_gif=True, path= date.strftime('%Y-%m-%d-bkg-nonan'),height=altura, background= bkg)
# kwgs = dict(height=altura,path='vlim',dates =instance.dataInfo.index[instance.dataInfo.index.hour <1 ])
kwgs = dict(
    height=.8,
    path= date.strftime('%m-%d'),
    textSave='',
    dates=instance.datos.index,
    makeGif=True,
)

instance.plot(output = 'P(r)',
    totalSignal=True,
    vlim=[0,135],
    parameters=['photon-s','photon-p'],
    scp=False,
    **kwgs )

instance.plot(output = 'P(r)',
    totalSignal=True,
    vlim=[5,34],
    parameters=['analog-s','analog-p'],
    scp=False,
    **kwgs )

instance.plot(
    parameters=['analog-s','analog-p'],
    vlim=[0.15,16], #[15,350]
    output='RCS',
    **kwgs
)

instance.plot(
    output='RCS',
    parameters=['analog-b'],
    vlim = [0.15,20],
    **kwgs
)



instance.plot(
    parameters=['photon-s','photon-p','photon-b'],
    vlim=[9,200], #[15,350]
    output='RCS',
    **kwgs
)

instance.plot(
    output='LVD',
    totalSignal=True,
    vlim = [0.25,1],
    **kwgs
)
kwgs['scp'] = False
instance.plot( output='Ln(RCS)', **kwgs )
# #

instance.plot(output='dLn(RCS)', **kwgs)

instance.plot(output='fLn(RCS)', **kwgs)

instance.plot(output='fdLn(RCS)',  **kwgs)

instance.plot(output='dfLn(RCS)',  **kwgs)

instance.plot(output='fdfLn(RCS)', **kwgs)


    except:
        pass
# instance.plot(
#     output='RCS',
#     parameters=['photon-b'],
#     vlim = [10,200],
#     **kwgs
# )
#### Informe de Actividades
#
# #
# #
# instance.plot( output='P(r)',**kwgs )
#
# instance.plot( output='RCS', **kwgs )
#
# instance.plot(textSave='_log', output='RCS',colorbarKind='Log',  **kwgs)
#
#
# instance.plot( output='Ln(RCS)', **kwgs )
# # #
#
# instance.plot(output='dLn(RCS)',colorbarKind='Anomaly',  **kwgs)
#
# instance.plot(scp=False,output='fLn(RCS)', **kwgs)
#
# instance.plot(output='fdLn(RCS)',colorbarKind='Anomaly',  **kwgs)
#
# instance.plot(output='dfLn(RCS)', colorbarKind='Anomaly', **kwgs)
#
# instance.plot(output='fdfLn(RCS)', colorbarKind='Anomaly',  **kwgs)

    # except:
    #     pass
# instance.plot(textSave='_test', parameters=['photon-p'], output='RCS', colorbarKind='Log', dates=instance.dataInfo.index, make_gif=True, path= '2018-07-04',scp=False)
# instance.plot(textSave='_test', parameters=['photon-p'], output='fdfLn(RCS)', colorbarKind='Anomaly', dates=instance.dataInfo.index, make_gif=True, path= '2018-07-04') #,vlim=[-2,4]
# instance.profiler(zenith=True,textSave='_RCS_log_test',parameter='analog-s',linear=False)
# instance.profiler(zenith=True,textSave='_RCS_log_test',parameter='analog-p',linear=False)
# instance.profiler(zenith=True,textSave='_RCS_log_test',parameter='photon-s',linear=False)
# instance.profiler(zenith=True,textSave='_RCS_log_test',parameter='photon-p',linear=False)
#
# instance.profiler(zenith=True,textSave='_RCS_log_test',parameter='photon-p',linear=False)

# plt.close('all')
# # # #  pd.concat({'ascii':ascii.data[ascii.data.columns.levels[0][1]].sort_index(axis=1), 'instance':instance.data[instance.data.columns.levels[0][1]].sort_index(axis=1)} , axis = 1)
# # #
# # ascii.data[ascii.data.columns.levels[0][1]]
# instance.get_output(output='RCS')
# x = instance.data[instance.data.columns.levels[0][1]]
# # x.mask(x==0,inplace=True)
# x.plot(xlim=(0,4000),subplots=True,figsize=(14,8),layout=(2,2),logy=True)
# plt.savefig('Figuras/Datos_Lidar_RCS.png',bbox_inches='tight' )
# os.system('scp Figuras/Datos_Lidar_RCS.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
# # #
#
# instance.data.mask(instance.data<=0,inplace=True)
# instance.data = np.log(instance.data)
# instance.profiler(zenith=True,textSave='_RCS_logdata_test',parameter='analog-s')
# instance.profiler(zenith=True,textSave='_RCS_logdata_test',parameter='analog-p')
# instance.profiler(zenith=True,textSave='_RCS_logdata_test',parameter='photon-s')
# instance.profiler(zenith=True,textSave='_RCS_logdata_test',parameter='photon-p')
#
#
# # #instance.data['2018-03-07 03:55:24'].iloc[:1200]
#
# # # #Plot
# plt.close('all')
# # # #  pd.concat({'ascii':ascii.data[ascii.data.columns.levels[0][1]].sort_index(axis=1), 'instance':instance.data[instance.data.columns.levels[0][1]].sort_index(axis=1)} , axis = 1)
# # #
# # ascii.data[ascii.data.columns.levels[0][1]]
# x = instance.data[instance.data.columns.levels[0][4]]
# # x.mask(x==0,inplace=True)
# x.plot(xlim=(0,4000),subplots=True,figsize=(14,8),layout=(2,2),logy=True)
# plt.savefig('Figuras/Datos_Lidar_RCS_log.png',bbox_inches='tight' )
# os.system('scp Figuras/Datos_Lidar_RCS_log.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
# #
# # # #
# # #
#
# #Plot
# plt.close('all')
# instance.data[instance.data.columns.levels[0][1]].plot(xlim=(0,4000),subplots=True,figsize=(14,8),layout=(2,2))
# plt.savefig('Figuras/Datos_Lidar_instance_derived_rezago18.png',bbox_inches='tight' )
# os.system('scp Figuras/Datos_Lidar_instance_derived_rezago18.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
# # # ################################################################################
# # # Regresion Mediana
# x   = ascii.data.groupby(axis=1,level=1).median()
# y   = instance.data.groupby(axis=1,level=1).median()
# z = pd.concat({'ascii':x,'instance':y},axis=1)
#
# plt.close('all')
# z.plot(xlim=(0,4000),subplots=True,figsize=(24,12),layout=(2,4))
# plt.savefig('Figuras/Datos_Lidar_media_derived.png',bbox_inches='tight')
# os.system('scp Figuras/Datos_Lidar_media_derived.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
#
# # for rezago in range(19):
# rezago=0
# plt.close('all')
# fig = plt.figure(figsize=(18,18))
# ax = {}
# for ix,col in enumerate(z.columns.levels[1].values):
#     print col,ix
#     # ax[ix] = fig.add_subplot(1 if ix <2 else 2,(ix%2)+1,(ix%2)+1)
#     ax[ix] = fig.add_subplot(2,2,ix+1)
#
#     # rl  = stats.linregress(x[col],y.loc[y.index [rezago:rezago-18]  if rezago<18 else y.index [18:],col])
#     rl  = stats.linregress(x[col],y[col])
#     title = 'slope = {} \nintercept = {} \nrvalue = {}'.format(rl.slope,rl.intercept,rl.rvalue)
#     # dataplot = z.xs(col,axis=1,level=1)
#     # ax[ix].scatter(x[col],y.loc[y.index [rezago:rezago-18]  if rezago<18 else y.index [18:],col],ylabel=col,)
#     ax[ix].scatter(x[col],y[col],ylabel=col,)
#     ax[ix].set_title(title)
#     ax[ix].legend()
# plt.savefig('Figuras/Datos_Lidar_RL_derived.png',bbox_inches='tight')
# os.system('scp Figuras/Datos_Lidar_RL_derived.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')


# from pandas.plotting import scatter_matrix
# plt.close('all')
# scatter_matrix(z, alpha=0.2, figsize=(24, 24), diagonal='kde')
# plt.savefig('Figuras/Datos_Lidar_median.scatter.png',bbox_inches='tight')
# os.system('scp Figuras/Datos_Lidar_median.scatter.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
#
# y.loc[y.index[:-18],['analog-p','analog-s']] = y.loc[y.index[18:],['analog-p','analog-s']].values
# y.loc[y.index[-18:],['analog-p','analog-s']] = np.NaN
# z = pd.concat({'ascii':x,'instance':y},axis=1)
#
# plt.close('all')
# scatter_matrix(z, alpha=0.2, figsize=(24, 24), diagonal='kde')
# plt.savefig('Figuras/Datos_Lidar_median.scatter-18.png',bbox_inches='tight')
# os.system('scp Figuras/Datos_Lidar_median.scatter-18.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
#
#
# ################################################################################
# # # # Multiple Linear Regression
#
# z = pd.concat({'ascii':ascii.data,'instance':instance.data})
#
# LR = pd.DataFrame() #columns =['slope','intercept','rvalue', 'pvalue', 'stderr'])
# for col in z.columns:
#     print col
#     y = z.loc['instance',col]
#     lr = stats.linregress(z.loc['ascii',col].values, (y.loc[y.index[18:]] if 'analog' in col[1] else y.loc[y.index[:-18]]).values  )
#     LR = LR.append( pd.DataFrame({'slope':lr.slope, 'intercept':lr.intercept, 'rvalue':lr.rvalue, 'pvalue':lr.pvalue, 'stderr':lr.stderr}, index=[col] ) )
#
# LR.index = pd.MultiIndex.from_tuples(LR.index)
# LR = LR.unstack()
#
#
# plt.close('all')
# plt.locator_params(axis='x', nbins=3)
# LR[['intercept','slope']].plot(kind='kde',subplots=True, layout=(2,4),figsize=(26,15), sharex=False )
# plt.savefig('Figuras/Datos_Lidar_Hist_LR.png',bbox_inches='tight')
# os.system('scp Figuras/Datos_Lidar_Hist_LR.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
# LR[['intercept','slope','rvalue']].max()
# LR[['intercept','slope','rvalue']].min()


#
# # ################################################################################
# #                 HISTORY
# # ################################################################################
# #
# Error = []
# os.system('mkdir Datos')
# dates = pd.date_range('20171219','20180522',freq='d')
#
# for d in dates:
# # d = dates[0]
#     os.system('rm -r Datos/*')
#     os.system('scp -r jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements/{}/* Datos/'.format( d.strftime('%Y%m%d')))
#
#     #Zenith
#     folders = glob.glob('Datos/ZS*')
#     if len(folders) > 0 :
#         os.system('ssh jhernandezv@siata.gov.co "mkdir /var/www/jhernandezv/Lidar/Zenith/{}/"'.format( d.strftime('%Y%m%d')))
#         for folder in folders:
#             files   = glob.glob('{}/RM*'.format(folder))
#             self    = Lidar(output='RCS')
#             self.read(files,)
#
#             try:
#                 self.plot(textSave = '_{}'.format(self.dataInfo.index[0].strftime('%H:%M')), path = 'jhernandezv/Lidar/Zenith/{}/'.format(d.strftime('%Y%m%d')),colorbarKind='Log')
#             except:
#                 Error.append(folder)
#                 pass
#
#     #Azimuth
#     folders = glob.glob('Datos/A*')
#     if len(folders) > 0 :
#         os.system('ssh jhernandezv@siata.gov.co "mkdir /var/www/jhernandezv/Lidar/Azimuth/{}/"'.format( d.strftime('%Y%m%d')))
#         for folder in folders:
#             files   = glob.glob('{}/RM*'.format(folder))
#             self    = Lidar(output='RCS')
#             self.read(files)
#             try:
#                 self.plot(zenith=False, colorbarKind='Log', \
#                     textSave = '_{}'.format(self.dataInfo.index[0].strftime('%H:%M')), \
#                     path = 'jhernandezv/Lidar/Azimuth/{}/'.format(d.strftime('%Y%m%d')))
#             except:
#                 Error.append(folder)
#                 pass
#
# ##############################################################################
# Error = []
# Zenith = {}
# #Fixed points
# # dates = pd.date_range('20171219','20180522',freq='d')
# dates = pd.date_range('20180425','20180522',freq='d')
# for d in dates:
# # d = dates[0]
#     os.system('rm -r Datos/*')
#     os.system('scp -r jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/LIDAR/Fixed_Point/{}/* Datos/'.format( d.strftime('%Y%m%d')))
#
#     files = glob.glob('Datos/RM*')
#     if len(files) > 0:
#         os.system('ssh jhernandezv@siata.gov.co "mkdir /var/www/jhernandezv/Lidar/FixedPoint/{}/"'.format( d.strftime('%Y%m%d')))
#         self = Lidar(scan=False,output='RCS')
#         self.read(files)
#
#         try:
#             Zenith [d.strftime('%Y%m%d')] = self.plot(colorbarKind='Log',zenith=False, \
#                 textSave = '_{}'.format(self.dataInfo.index[0].strftime('%H:%M')), \
#                 path = 'jhernandezv/Lidar/FixedPoint/{}/'.format(d.strftime('%Y%m%d')))
#         except:
#             Error.append(d.strftime('%Y%m%d'))
#             pass
#
# print Zenith
# ###############################################################################
# #3Ds
# os.system('rm Figuras/*')
# os.system('mkdir Datos')
# # dates = pd.date_range('20180624','20180717',freq='d')
# dates = pd.date_range('20180704','20180705',freq='d')
#
# for d in dates:
# # d = dates[0]
#     os.system('rm -r Datos/*')
#     os.system('scp -r jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/LIDAR/Scanning_Measurements/{}/* Datos/'.format( d.strftime('%Y%m%d')))
#
#     folders = glob.glob('Datos/3D*')
#     if len(folders) > 0 :
#         os.system('ssh jhernandezv@siata.gov.co "mkdir /var/www/jhernandezv/Lidar/3D/{}/"'.format( d.strftime('%Y%m%d')))
#         for folder in folders:
#             files   = glob.glob('{}/RM*'.format(folder))
#             self    = Lidar(output='RCS',ascii=d.strftime('%Y-%m-%d') in ['2018-06-24','2018-06-25','2018-06-26'] )
#             print folder
#             self.read(files,tresd=True)
#             try:
#                 self.plot(colorbarKind='Log', textSave ='_{}'.format(self.dataInfo.index[0].strftime('%H:%M')), \
#                         path= '{}3D/{}/'.format(self.kwargs['path'],d.strftime('%Y%m%d')), vlim=[6.7,9])
#             except:
#                 Error.append(folders)
#                 pass
#         # os.system('convert -delay 20 -loop 0 {}*.png {}.gif'.format())
#         for col in self.data.columns.levels[1].values:
#             os.system( 'convert -delay 20 -loop 0 Figuras/Lidar_Scanning_RCS_{}_* Figuras/lidar_Scanning_RCS_{}_{}.gif'.format(col,col,d.strftime('%Y%m%d')))
#             os.system('scp Figuras/lidar_Scanning_RCS_{}_{}.gif jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/3D/{}/ '.format(col,d.strftime('%Y%m%d'),d.strftime('%Y%m%d')) )

class Employee(object):

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    def testmethod(self,any):
        print 'Test 1'



# class Developer(Employee):
#
#     raise_amt = 1.10
#
#     def __init__(self, test):
#         self.test = test
#
#     def set_dev(self,first, last, pay, prog_lang):
#         super(Developer,self).__init__(first, last, pay)
#         self.prog_lang = prog_lang
class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super(Developer,self).__init__(first, last, pay)
        self.prog_lang = prog_lang






class Manager(Developer):
    raise_amt = 1.2

    def __init__(self, first, last, pay,prog_lang='python', employees=None):
        super(Manager,self).__init__(first, last, pay, prog_lang)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())

    def testmethod(self,any):
        print 'Test 2'

dev_1 = Developer('Corey', 'Schafer', 50000, 'Python')
dev_2 = Developer('Test', 'Employee', 60000, 'Java')

mgr_1 = Manager('Sue', 'Smith', 90000, 'Python', [dev_1])

print(mgr_1.email)

mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_1)

mgr_1.print_emps()
