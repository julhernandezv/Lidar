# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('PDF')
import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime as dt
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
# ascii = Lidar(ascii=True,scan=False,output='raw_data')
# ascii.read(files,inplace=False)
# # #
# #
# # # ################################################################################
# # # Binario
# files = glob.glob('InfoLidar/AS0_180307-035517/RM*')
# files = glob.glob('InfoLidar/ZS0_180223-131945/RM*')
# files = glob.glob('InfoLidar/ZS0_180709-151714/RM*')
#
#
# binario = Lidar(scan=True,output='RCS')
# binario.read(files)
#
# plt.close('all')
# binario.data.groupby(axis=1,level=1).median().boxplot(column=['analog-p','analog-s'])
# binario._save_fig(textsave='_boxplot_median_analog')
# plt.close('all')
# binario.data.groupby(axis=1,level=1).median().boxplot(column=['photon-p','photon-s'])
# binario._save_fig(textsave='_boxplot_median_photon')

# binario.data.quantile(np.arange(0,1,0.01)).groupby(axis=1,level=1).median()



# files = glob.glob('InfoLidar/ZS0_180709-151714/RM*')      # test
# files = glob.glob('InfoLidar/3Ds_180703-135028/RM*')        # test2
# files = glob.glob('InfoLidar/3Ds_180704-105519/RM*')        # test3

# binario = Lidar(Fechai='2018-07-04',Fechaf='2018-07-04',scan='3D')
# binario.read()

# backup = [binario.data, binario.data_info]
# binario.data        = backup[0]
# binario.raw_data    = backup[0]
# binario.data_info   = backup[1]
#
# binario.plot(textsave='_test5_',parameters=['photon-p'])
# binario.plot(textsave='_test5_log',parameters=['photon-p'],output='RCS',colorbar_kind='Log')
# binario.plot(textsave='_test_4D',parameters=['photon-p'],output='RCS')
# # dd, di = binario.read_folder(files)
# # '-75.5686', '6.2680'io.data.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])

#
# binario.plot(textsave='_test_1',parameters=['photon-p'])
# binario.plot(textsave='_test_1',parameters=['photon-p'],output='RCS')
# binario.plot(textsave='_log_test_1',output='RCS',parameters=['photon-p'],colorbar_kind='Log')
# binario.plot(textsave='_test_1',output='Ln(RCS)',parameters=['photon-p'])
# binario.plot(textsave='_test_1',output='dLn(RCS)',parameters=['photon-p'],colorbar_kind='Anomaly')
# binario.plot(textsave='_test_1',output='fLn(RCS)',parameters=['photon-p'])
# binario.plot(textsave='_test_1',output='fdLn(RCS)',parameters=['photon-p'],colorbar_kind='Anomaly')
# binario.plot(textsave='_test_1',output='dfLn(RCS)',parameters=['photon-p'],colorbar_kind='Anomaly')
# binario.plot(textsave='_test_1',output='fdfLn(RCS)',parameters=['photon-p'],colorbar_kind='Anomaly')

################################################################################
# FixedPoint#
# date = pd.date_range('2018-08-06','2018-08-06',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
# altura = 4.5
# binario = Lidar(Fechai=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='FixedPoint')
# binario.read()
# binario.data = binario.data.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])
# binario.raw_data = binario.data
# binario.data_info = binario.data_info.resample('30s').mean()
# binario.data_info = binario.data_info.reindex( pd.date_range(binario.data.columns.levels[0][0],binario.data.columns.levels[0][-1],freq='30s'))
# kwgs = dict( height=altura,)# background= bkg)
# binario.plot(**kwgs )

# binario.data.reindex( pd.date_range(binario.data.columns[0],binario.data.columns[-1],freq='30s'), axis=1)
# binario.data.reindex(pd.date_range(binario.data.columns.levels[0][0],binario.data.columns.levels[0][-1],freq='30s'))
#################################################################################
################################################################################
################################################################################
# FixedPoint#
# for date in pd.date_range('2018-08-01','2018-08-07',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
    # try:
date = pd.date_range('2018-08-03','2018-08-03',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
#
binario = Lidar(Fechai=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='FixedPoint',output='raw_data')
binario.read()
binario.datos = binario.datos.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])
binario.raw_data = binario.datos.copy()
binario.datos_info = binario.datos_info.resample('30s').mean()
#
backup = [binario.datos, binario.datos_info]
# binario.datos        = backup[0]
# binario.raw_data    = backup[0].copy()
# binario.datos_info   = backup[1]
#
# #
kwgs = dict( height=4.5, path='informe',textsave='Test_MPLPlot')
binario.plot(output = 'P(r)',**kwgs )
#
# binario.plot( output='RCS', **kwgs )
#
# binario.plot(textsave='_log' + kwgs.pop('textsave'), output='RCS',colorbar_kind='Log',  **kwgs)
#
#
# binario.plot( output='Ln(RCS)', **kwgs )
################################################################################
## Ceilomoetro
# from Funciones_Lectura import lee_Ceil,lee_data_ceil
# # import pandas as pd
# # #
# fecha_test = pd.date_range('2018-08-03 09:35:00', '2018-08-03 18:56:30',freq='30s')#binario.data.columns.levels[0]
# ceil =lee_data_ceil('amva',fecha_test[0].strftime('%Y-%m-%d %H:%M:%S'),fecha_test[-1].strftime('%Y-%m-%d %H:%M:%S') ) #fecha_test[0].strftime('%Y%m%d %H:%M'),fecha_test[-1].strftime('%Y%m%d %H:%M'),'amva')
# ceil.columns = (ceil.columns+1)/100.
# ceil[ceil < 100] = 100
# ceil[ceil.isnull()] = 50
################################################################################# binario.plot_lidar(ceil.index,ceil.columns,ceil.T,textsave='Ceilometro_AMVA',path='jhernandezv/Lidar/FixedPoint/',colorbar_kind='Log',vlim=np.log10([10,13000]),ylabel='Intensidad Backscatter $[10^{-9}m^{-1}sr^{-1}]$')

        #

        # binario.plot(output='dLn(RCS)',colorbar_kind='Anomaly',  **kwgs)

        # binario.plot(output='fLn(RCS)', **kwgs)

        # binario.plot(output='fdLn(RCS)',colorbar_kind='Anomaly',  **kwgs)

        # binario.plot(output='dfLn(RCS)', colorbar_kind='Anomaly', **kwgs)

        # binario.plot(output='fdfLn(RCS)', colorbar_kind='Anomaly',  **kwgs)

        #10 km
        # kwgs = dict(  height=10,textsave='_10km',path='10km')# background= bkg)
        #
        # binario.plot(output = 'P(r)', **kwgs )
        #
        # binario.plot( output='RCS', **kwgs )
        #
        #
        # binario.plot( output='Ln(RCS)', **kwgs )
        # # #
        #
        # # binario.plot(output='dLn(RCS)',colorbar_kind='Anomaly',  **kwgs)
        #
        # binario.plot(output='fLn(RCS)', **kwgs)
        #
        # # binario.plot(output='fdLn(RCS)',colorbar_kind='Anomaly',  **kwgs)
        #
        # # binario.plot(output='dfLn(RCS)', colorbar_kind='Anomaly', **kwgs)
        #
        # binario.plot(output='fdfLn(RCS)', colorbar_kind='Anomaly',  **kwgs)
        #
        # kwgs.pop('textsave')
        # binario.plot(textsave='_log_10km', output='RCS',colorbar_kind='Log',  **kwgs)


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
#### Informe de Actividades
# altura = 4.5
# # # for date in pd.date_range('2018-06-30','2018-06-30',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
# # # try:
# date = pd.date_range('2018-06-30','2018-06-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
#
# binario = Lidar(Fechai=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='3D')
# binario.read()
# # backup = [binario.raw_data, binario.data_info]
# # binario.data        = backup[0]
# # binario.raw_data    = backup[0]
# # binario.data_info   = backup[1]
#
#
# # kwgs = dict(parameters=['photon-p'], dates=binario.data_info.index, make_gif=True, path= date.strftime('%Y-%m-%d-bkg-nonan'),height=altura, background= bkg)
# kwgs = dict(height=altura,path='informe',dates =binario.data_info.index[binario.data_info.index.hour <1 ])
#
# #
# #
# binario.plot( output='P(r)',**kwgs )
#
# binario.plot( output='RCS', **kwgs )
#
# binario.plot(textsave='_log', output='RCS',colorbar_kind='Log',  **kwgs)
#
#
# binario.plot( output='Ln(RCS)', **kwgs )
# # #
#
# binario.plot(output='dLn(RCS)',colorbar_kind='Anomaly',  **kwgs)
#
# binario.plot(scp=False,output='fLn(RCS)', **kwgs)
#
# binario.plot(output='fdLn(RCS)',colorbar_kind='Anomaly',  **kwgs)
#
# binario.plot(output='dfLn(RCS)', colorbar_kind='Anomaly', **kwgs)
#
# binario.plot(output='fdfLn(RCS)', colorbar_kind='Anomaly',  **kwgs)

    # except:
    #     pass
# binario.plot(textsave='_test', parameters=['photon-p'], output='RCS', colorbar_kind='Log', dates=binario.data_info.index, make_gif=True, path= '2018-07-04',scp=False)
# binario.plot(textsave='_test', parameters=['photon-p'], output='fdfLn(RCS)', colorbar_kind='Anomaly', dates=binario.data_info.index, make_gif=True, path= '2018-07-04') #,vlim=[-2,4]
# binario.profiler(zenith=True,textsave='_RCS_log_test',parameter='analog-s',linear=False)
# binario.profiler(zenith=True,textsave='_RCS_log_test',parameter='analog-p',linear=False)
# binario.profiler(zenith=True,textsave='_RCS_log_test',parameter='photon-s',linear=False)
# binario.profiler(zenith=True,textsave='_RCS_log_test',parameter='photon-p',linear=False)
#
# binario.profiler(zenith=True,textsave='_RCS_log_test',parameter='photon-p',linear=False)

# plt.close('all')
# # # #  pd.concat({'ascii':ascii.data[ascii.data.columns.levels[0][1]].sort_index(axis=1), 'binario':binario.data[binario.data.columns.levels[0][1]].sort_index(axis=1)} , axis = 1)
# # #
# # ascii.data[ascii.data.columns.levels[0][1]]
# binario.derived_output(output='RCS')
# x = binario.data[binario.data.columns.levels[0][1]]
# # x.mask(x==0,inplace=True)
# x.plot(xlim=(0,4000),subplots=True,figsize=(14,8),layout=(2,2),logy=True)
# plt.savefig('Figuras/Datos_Lidar_RCS.png',bbox_inches='tight' )
# os.system('scp Figuras/Datos_Lidar_RCS.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
# # #
#
# binario.data.mask(binario.data<=0,inplace=True)
# binario.data = np.log(binario.data)
# binario.profiler(zenith=True,textsave='_RCS_logdata_test',parameter='analog-s')
# binario.profiler(zenith=True,textsave='_RCS_logdata_test',parameter='analog-p')
# binario.profiler(zenith=True,textsave='_RCS_logdata_test',parameter='photon-s')
# binario.profiler(zenith=True,textsave='_RCS_logdata_test',parameter='photon-p')
#
#
# # #binario.data['2018-03-07 03:55:24'].iloc[:1200]
#
# # # #Plot
# plt.close('all')
# # # #  pd.concat({'ascii':ascii.data[ascii.data.columns.levels[0][1]].sort_index(axis=1), 'binario':binario.data[binario.data.columns.levels[0][1]].sort_index(axis=1)} , axis = 1)
# # #
# # ascii.data[ascii.data.columns.levels[0][1]]
# x = binario.data[binario.data.columns.levels[0][4]]
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
# binario.data[binario.data.columns.levels[0][1]].plot(xlim=(0,4000),subplots=True,figsize=(14,8),layout=(2,2))
# plt.savefig('Figuras/Datos_Lidar_Binario_derived_rezago18.png',bbox_inches='tight' )
# os.system('scp Figuras/Datos_Lidar_Binario_derived_rezago18.png jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/')
#
# # # ################################################################################
# # # Regresion Mediana
# x   = ascii.data.groupby(axis=1,level=1).median()
# y   = binario.data.groupby(axis=1,level=1).median()
# z = pd.concat({'ascii':x,'binario':y},axis=1)
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
# z = pd.concat({'ascii':x,'binario':y},axis=1)
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
# z = pd.concat({'ascii':ascii.data,'binario':binario.data})
#
# LR = pd.DataFrame() #columns =['slope','intercept','rvalue', 'pvalue', 'stderr'])
# for col in z.columns:
#     print col
#     y = z.loc['binario',col]
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
#                 self.plot(textsave = '_{}'.format(self.data_info.index[0].strftime('%H:%M')), path = 'jhernandezv/Lidar/Zenith/{}/'.format(d.strftime('%Y%m%d')),colorbar_kind='Log')
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
#                 self.plot(zenith=False, colorbar_kind='Log', \
#                     textsave = '_{}'.format(self.data_info.index[0].strftime('%H:%M')), \
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
#             Zenith [d.strftime('%Y%m%d')] = self.plot(colorbar_kind='Log',zenith=False, \
#                 textsave = '_{}'.format(self.data_info.index[0].strftime('%H:%M')), \
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
#                 self.plot(colorbar_kind='Log', textsave ='_{}'.format(self.data_info.index[0].strftime('%H:%M')), \
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
