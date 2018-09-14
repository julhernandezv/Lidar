# -*- coding: utf-8 -*-
from matplotlib import use, cm, colors
use('PDF')

import datetime as dt
import numpy as np


import pandas as pd
import struct
import sys, os, glob, locale

from .plotbook import PlotBook
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import register_cmap
from matplotlib.dates import DateFormatter
# reload (sys)
# sys.setdefaultencoding ("utf-8")
locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))



#
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    cdict = {'red': [],'green': [],'blue': [],'alpha': []}

    reg_index = np.linspace(start, stop, 257)
#
    shift_index = np.hstack([

        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)

    register_cmap(cmap=newcmap)
    return newcmap

shrunk_cmap = shiftedColorMap(cm.jet, start=0.15,midpoint=0.45, stop=0.85, name='shrunk')
shrunk_cmap2 = shiftedColorMap(cm.jet, start=0.0,midpoint=0.65, stop=0.85, name='shrunk_ceil')


class Lidar(PlotBook):
    """
    Class for manipulating SIATA's Scanning Lidar

    Parameters
        output      = 'raw_data','P(r)', 'S(r)', 'Depolarization' 'RCS','Ln(RCS)','fLn(RCS)','dLn(RCS)','fdLn(RCS)','dfLn(RCS)','fdfLn(RCS)'... - options for derived outputs
        scan        = ['Zenith','Azimuth','3D','FixedPoint'] - kind of spacial measurement
        ascii       = bolean - if False read binary files

    """

    lidar_props   = {
                'raw_data': {'analog':'',
                    'photon':'',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'P(r)': {'analog':r'$[mV]$',
                    'photon':r'$[MHz]$',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'S(r)': {'analog':r'$[mV]$',
                    'photon':r'$[MHz]$',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'Depolarization': {'analog':r'Depolarization',
                    'photon':r'Depolarization',
                    'cmap':cm.seismic,
                    'colorbar_kind':'Anomaly'},
                'RCS': {'analog':r'RCS $[mV*Km^2]$',
                    'photon':r'RCS $[MHz*Km^2]$',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'Ln(RCS)': {'analog':r'Ln(RCS) $[Ln(mV*Km^2)]$',
                    'photon':r'Ln(RCS) $[Ln(MHz*Km^2)]$',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'fLn(RCS)':{'analog':r'fLn(RCS) $[Ln(mV*Km^2)]$',
                    'photon':r'fLn(RCS) $[Ln(MHz*Km^2)]$',
                    'cmap':shrunk_cmap,
                    'colorbar_kind':'Linear'},
                'dLn(RCS)':{'analog':r'dLn(RCS)',
                    'photon':r'dLn(RCS)',
                    'cmap':cm.seismic,
                    'colorbar_kind':'Anomaly'},
                'fdLn(RCS)':{'analog':r'fdLn(RCS)',
                    'photon':r'fdLn(RCS)',
                    'cmap':cm.seismic,
                    'colorbar_kind':'Anomaly'},
                'dfLn(RCS)':{'analog':r'dfLn(RCS)',
                    'photon':r'dfLn(RCS)',
                    'cmap':cm.seismic,
                    'colorbar_kind':'Anomaly'},
                'fdfLn(RCS)':{'analog':r'fdfLn(RCS)',
                    'photon':r'fdfLn(RCS)',
                    'cmap':cm.seismic,
                    'colorbar_kind':'Anomaly'},
    }

    ceil_cmap = shrunk_cmap2

    bkg = {'analog-p':5.29, 'analog-s':4.984, 'photon-p':0.13289, 'photon-s':0.13289} #0.26578

    V0 = 0.051


    def __init__(self, Fechai=None, Fechaf=None, ascii=False, scan='3D', output='P(r)', **kwargs):

        self.ascii      = ascii
        self.scan       = scan
        self.output     = output
        self.Fechai     = (dt.datetime.now() - relativedelta(months=1)
                        ).strftime('%Y-%m-') + '01 01:00' if (Fechaf == None) else Fechai
        self.Fechaf     = (pd.to_datetime(self.Fechai)+ relativedelta(months=1
                            ) - dt.timedelta(hours=1)
                        ).strftime('%Y-%m-%d %H:%M') if (Fechaf == None) else Fechaf #

        self.degree_variable    = 'Zenith' if self.scan in ['FixedPoint','3D'] else self.scan
        self.degree_fixed       = 'Azimuth'  if self.scan in ['FixedPoint','3D','Zenith'] else 'Zenith'
        self.kwargs             = kwargs

        # super()


    def read_file(self,filename):
        """Function for reading Lidar files

        Parameters
            filename    = str - path to file

        Return
            dataset     = pandas DataFrame object - parametrization by column, spacial range on index
            description = pandas DataFrame object - metadata by dataset
        """
        fileObj = open (filename, "rb")


        lineaMeasurementName = fileObj.readline ()
        # lineaMeasurementName = lineaMeasurementName.strip ()
        print lineaMeasurementName
        print "***********************************************************************************************************************************************"

        lineaLocation = fileObj.readline ()
        lineaLocationArray = lineaLocation.strip ().split()
        print lineaLocationArray

        description = {}

        # ValorLocationStr = lineaLocationArray[0]
        description ['Fecha'] = dt.datetime.strptime (lineaLocationArray[1] + " " + lineaLocationArray[2], "%d/%m/%Y %H:%M:%S") - dt.timedelta(hours=5)
        description ['Fecha_fin'] = dt.datetime.strptime (lineaLocationArray[3] + " " + lineaLocationArray[4], "%d/%m/%Y %H:%M:%S") - dt.timedelta(hours=5)
        # valorHeight = float (lineaLocationArray[5])
        # valorLong = float (lineaLocationArray[6])
        # valorLat = float (lineaLocationArray[7])
        description ['Zenith'] = int( float (lineaLocationArray[8]) )
        description ['Azimuth'] = int( float (lineaLocationArray[9]) )
        # description ['Temp'] = float (lineaLocationArray[10])
        # description ['Press'] = float (lineaLocationArray[11])
#
        print description
        print "***********************************************************************************************************************************************"

        # The third line contains information about the lidar’s offset from the North (En el file que descargue de FixedPoint no estaba esta linea - en Scan si)
        if self.scan not in ['FixedPoint']:
            lineaOffsetNorth = fileObj.readline ()
            lineaOffsetNorth = lineaOffsetNorth.strip ()
            print lineaOffsetNorth
            print "***********************************************************************************************************************************************"

        lineaInfoLaser = fileObj.readline ()
        lineaInfoLaserArray = lineaInfoLaser.strip ().split ()
        print lineaInfoLaserArray

        description ['NumberShotsLaser1'] = int (lineaInfoLaserArray[0])
        description ['PulseRepFreqLaser1'] = int (lineaInfoLaserArray[1])
        # description ['NumberShotsLaser2'] = int (lineaInfoLaserArray[2])
        # description ['PulseRepFreqLaser2'] = int (lineaInfoLaserArray[3])
        valorNumDatasets = int (lineaInfoLaserArray[4])

        print "***********************************************************************************************************************************************"

        dictDescripcionDataset = {}

        for idiDataset in xrange (valorNumDatasets):

            lineaInfoDataset = fileObj.readline ()
            lineaInfoDatasetArray = lineaInfoDataset.strip ().split ()
            print lineaInfoDatasetArray


            dictDescripcionDataset[idiDataset + 1] = {}
            #
            # dictDescripcionDataset[idiDataset + 1]["datasetPresent"] = (int (lineaInfoDatasetArray[0]) == 1)
            dictDescripcionDataset[idiDataset + 1]["datasetModoAnalogo"]    = (int (lineaInfoDatasetArray[1]) == 0)
            # dictDescripcionDataset[idiDataset + 1]["datasetModoPhotonCount"] = (int (lineaInfoDatasetArray[1]) == 1)
            # dictDescripcionDataset[idiDataset + 1]["datasetLaserNumber"] = int (lineaInfoDatasetArray[2])
            dictDescripcionDataset[idiDataset + 1]["datasetBinNums"]        = int (lineaInfoDatasetArray[3])
            # dictDescripcionDataset[idiDataset + 1]["datasetNaDigit"] = int (lineaInfoDatasetArray[4])
            # dictDescripcionDataset[idiDataset + 1]["datasetPMTHighVoltage"] = float (lineaInfoDatasetArray[5])
            dictDescripcionDataset[idiDataset + 1]["datasetBinWidth"]       = float (lineaInfoDatasetArray[6])
            # dictDescripcionDataset[idiDataset + 1]["datasetLaserWavelength"] = lineaInfoDatasetArray[7][:5]
            dictDescripcionDataset[idiDataset + 1]["datasetPolarization"]   = lineaInfoDatasetArray[7][6]

            parameter = "{}-{}".format('analog' if dictDescripcionDataset[idiDataset + 1]["datasetModoAnalogo"] else 'photon', dictDescripcionDataset[idiDataset + 1]["datasetPolarization"])

            if lineaInfoDatasetArray[15][0:2] == 'BT': #dictDescripcionDataset[idiDataset + 1]["datasetModoAnalogo"]:
                print int (lineaInfoDatasetArray[12]),dictDescripcionDataset[idiDataset + 1]["datasetPolarization"]
                description ['ADCBits_'+parameter]      = int (lineaInfoDatasetArray[12])
                description ['InputRange_'+parameter]   = float (lineaInfoDatasetArray[14])
                description ['ShotNumber_'+parameter]   = int (lineaInfoDatasetArray[13])

            elif lineaInfoDatasetArray[15][0:2] == 'BC':
                description ['ShotNumber_'+parameter]   = int (lineaInfoDatasetArray[13])
                description ['BinWidth_'+parameter]     = float (lineaInfoDatasetArray[6])
                # dictDescripcionDataset[idiDataset + 1]["datsetInputRange"] = float (lineaInfoDatasetArray[14][:5])
        #         dictDescripcionDataset[idiDataset + 1]["datsetDiscriminatorlevel"] = int (lineaInfoDatasetArray[14][5:])

            dictDescripcionDataset[idiDataset + 1]["datasetDescriptor"] = lineaInfoDatasetArray[15][0:2]
            # dictDescripcionDataset[idiDataset + 1]["datasetHexTransientRecNum"] = int (lineaInfoDatasetArray[15][-1], 16)

            print dictDescripcionDataset[idiDataset + 1]

        print "***********************************************************************************************************************************************"

        # The dataset description is followed by an extra CRLF.
        print 'CRLF after description = {}'.format(fileObj.readline ())

        if self.ascii:
            fileObj.close ()
            dataset =   pd.read_csv(filename,
                            delimiter='\t',
                            header=9 if self.scan not in ['FixedPoint'] else 8,
                            usecols=[0,1,2,3]
                        )
            ejex =      np.array(
                            range(1,dataset.shape[0]+ 1)
                        )*dictDescripcionDataset[1]["datasetBinWidth"] / 1000.

            dataset.index  = ejex
            dataset.columns = dataset.columns.str.strip('355.000 .').str.strip(' 0 ').str.strip(' 1 ').str.replace(' ','-')
            dataset.columns = dataset.columns.str.slice(2) +'-'+ dataset.columns.str.get(0)

        else:
            # The datasets are stored as 32bit integer values. Datasets are separated by CRLF. The last dataset is followed by a CRLF.
            # These CRLF are used as markers and can be used as check points for file integrity.
            dataset = []
            for ix in range (1, valorNumDatasets+1):
                if dictDescripcionDataset[ix]["datasetDescriptor"] in ['BT','BC']:

                    dictDescripcionDataset[ix]["datasetLista"] = []

                    for idiBin in xrange (dictDescripcionDataset[ix]["datasetBinNums"]):

                        dictDescripcionDataset[ix]["datasetLista"].append ((struct.unpack ('i', fileObj.read (4)))[0])

                    print 'CRLF integrity {} = {}'.format(dictDescripcionDataset[ix]["datasetDescriptor"],fileObj.readline ())

                    ejex = np.array(range(1,dictDescripcionDataset[ix]["datasetBinNums"]-17)) * dictDescripcionDataset[ix]["datasetBinWidth"] / 1000.
                    dataset.append( pd.DataFrame(dictDescripcionDataset[ix]['datasetLista'][18:] if dictDescripcionDataset[ix]["datasetDescriptor"] == 'BT' else dictDescripcionDataset[ix]['datasetLista'][:-18], index=ejex, columns=[ "{}-{}".format('analog' if dictDescripcionDataset[ix]["datasetModoAnalogo"] else 'photon', dictDescripcionDataset[ix]["datasetPolarization"])] ))

                    # print dictDescripcionDataset[ix]

            print "***********************************************************************************************************************************************"

            dataset             = pd.concat(dataset,axis=1)
            dataset.sort_index(axis=1,inplace=True)
            dataset.index.name  = 'Height'
            fileObj.close ()
            description = pd.DataFrame(description,index=[1]).set_index('Fecha')


        return dataset, description

    def read_folder(self,filenames,**kwargs):
        """Function for reading lidar files (binary or ascii)

        Parameters
            filenames   = [] - List of files

            **kwargs
            inplace  = bolean, return values insted using heritance

        """

        data_info   = pd.DataFrame()
        data        = {}

        for file in sorted(filenames):
            print "{} \n {} \n {}".format("="*50,file,"="*50)
            df1, df2    = self.read_file(file)
            # print df2.index.strftime('%Y-%m-%d %H:%M:%S')
            # print '\n index= {}'.format(df2['Zenith' if self.scan in ['FixedPoint','3D'] else self.scan].values[0])
            data[ df2.index[0].strftime('%Y-%m-%d %H:%M:%S') ] = df1
            data_info   = data_info.append(df2)

        data        = pd.concat(data,axis=1)
        data_info   = data_info.sort_index() #.reset_index()
        data.columns.set_levels( pd.to_datetime(data.columns.levels[0].values), level=0, inplace=True) #range(data.columns.levels[0].size), level=0, inplace=True)

        if self.scan == '3D':
            if np.abs(data_info.Zenith.iloc[1] - data_info.Zenith.iloc[0]) > 10:
                print data_info.index[0]
                # data_info.loc[data_info.index[0],'Zenith'] = data_info.Zenith.iloc[1] + 5
                data.drop(data_info.index[0],axis=1,level=0,inplace=True)
                data_info.drop(data_info.index[0],inplace=True)

            data_info.loc[ data_info.Azimuth != data_info.Azimuth.iloc[0], ['Azimuth','Zenith'] ] += 180
            data_info.loc[ data_info.Azimuth == data_info.Azimuth.iloc[0], 'Zenith' ] *= -1

            #Filter by repeated measures at Zenith 90
            duplicated =  data_info.index[data_info.Zenith == 90]
            print duplicated#duplicated('Zenith',keep=False)

            if duplicated.size >1 :
                print "\n Deleting duplicated"
                data.loc[:, duplicated[0] ] = data[ duplicated ].groupby(
                    axis=1,
                    level=1).mean().values
                data.drop(
                    duplicated[1:],
                    axis=1,
                    level=0,
                    inplace=True)
                data_info.drop(
                    duplicated[1:],
                    inplace=True)

        elif self.scan in ['Zenith','Azimuth','FixedPoint']:
            data_info.loc[:,'Zenith']   *= -1

        data_info['Azimuth']    = (270-data_info['Azimuth'])%360

        data.columns    = pd.MultiIndex.from_product(
            [ data_info[ self.degree_variable ].values, data.columns.levels[-1].values ],
            names = [self.degree_variable,'Parameters'] )

        #Filtro para mediciones inferiores a 110m de distacia al sensor
        # data       = data[data.index >= 0.110]

        if kwargs.get('inplace',True):
            self.datos       = data
            self.datos_info  = data_info
            self.raw_data       = self.datos.copy()
            self.derived_output(
                output=kwargs.pop('output',None),
                total_signal=kwargs.pop('total_signal',False)
            )

        else:
            return data, data_info



    def read(self, **kwargs):

        kind_folder = {'3D':'3D','Zenith':'Z','Azimuth':'A','FixedPoint':'RM'}
        ## os.system('rm Figuras/*')
        os.system('mkdir Datos')
        # dates = pd.date_range('20180624','20180717',freq='d')
        dates = pd.date_range(self.Fechai,self.Fechaf,freq='d')
        print dates

        self.datos       = {}
        self.datos_info  = pd.DataFrame()

        for d in dates:
        # d = dates[0]
            os.system('rm -r Datos/*')
            os.system('scp -r jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/LIDAR/{}/{}/* Datos/'.format('Scanning_Measurements' if self.scan != 'FixedPoint' else 'Fixed_Point', d.strftime('%Y%m%d')))

            folders = glob.glob('Datos/{}*'.format( kind_folder[self.scan]))
            if len(folders) > 0 :
                # os.system('ssh jhernandezv@siata.gov.co "mkdir /var/www/jhernandezv/Lidar/{}/{}/"'.format(self.scan, d.strftime('%Y%m%d')))

                for folder in folders:
                    archivos   = glob.glob('{}{}*'.format(folder, '/RM' if self.scan != 'FixedPoint' else ''))
                    print folder
                    df3, df4 = self.read_folder(archivos, inplace=False)
                    self.datos[ df4.index[0].strftime('%Y-%m-%d %H:%M:%S') ] = df3.stack([0,1])

                    df4.loc[df4.index[0], 'Fecha_fin'] = df4.index[-1]
                    self.datos_info = self.datos_info.append(df4.iloc[0])

        self.datos           = pd.concat(self.datos,axis=1).T
        self.datos.index     = pd.to_datetime( self.datos.index )
        self.datos_info.sort_index(inplace=True)
        self.raw_data       = self.datos.copy()
        self.derived_output(
            output=kwargs.pop('output',None),
            total_signal=kwargs.pop('total_signal',False)
        )


    def derived_output(self,output=None,total_signal=False):
        """Method for derived values
            Parameters
                output = Allowed {}
                total_signal = Bolean, default False. Used to get depolarization signal """.format(self.lidar_props.keys())


        if output is not None:

            self.output     = output

        if self.output in self.lidar_props.keys():
            self.datos       = self.raw_data.copy()

            if self.output not in ['raw_data']:
                self.datos       = self.Pr

                if self.output not in ['P(r)']:
                    self.datos      = self.background

                    if self.output == 'Depolarization':
                        self.datos = self.depolarization_ratio

                    elif total_signal:
                        self.datos = self.total_signal



            if self.output in ['RCS', 'Ln(RCS)', 'fLn(RCS)',
                        'dLn(RCS)', 'fdLn(RCS)', 'dfLn(RCS)', 'fdfLn(RCS)']:

                self.datos      = self.RCS

                if self.output not in ['RCS']:
                    # self.datos.mask(self.datos<=0,inplace=True)
                    self.datos[ self.datos <= 0.1 ] = 0.1
                    self.datos       = np.log(self.datos)

                if self.output in ['fLn(RCS)','dfLn(RCS)','fdfLn(RCS)']:
                    self.datos       = self.average_filter

                if self.output in ['dLn(RCS)','fdLn(RCS)','fdfLn(RCS)','dfLn(RCS)']:
                    self.datos       = self.derived

                if self.output in ['fdLn(RCS)','fdfLn(RCS)']:
                    self.datos       = self.average_filter

        else:
            print "Output '{}' not allowed, check other".format(self.output)


    def _make_plot(self):
        cax     = self.fig.add_axes((1.02,.2,0.02,0.59))
        self.axes[0].patch.set_facecolor((.75,.75,.75))
        self._make_contour(cax = cax)


    def plot_lidar(self,X,Y,Z,**kwargs):
        """Function for ploting lidar profiles

        Parameters
            X,Y     = one or two dimensional array for grid plot
            Z       = DataFrame object for contour

            **PlotBook kwargs allowed
        """

        if self.scan not in ['FixedPoint']:
            rel                 = (Y.max()-Y.min())/(np.abs(X).max()-X.min())
            kwargs['figsize']   = ( 10,10*rel) if rel <=1 else (10*(1./rel),10)
            kwargs['xlim']      = [-np.abs(X).max(), np.abs(X).max()]
            kwargs['xlabel']    = r'Range $[Km]$'
        else:
            kwargs['figsize']   = (10,5.6)

        kwargs['ylabel']        = r'Range $[Km]$'
        kwargs['colormap']      = kwargs.get('colormap',self.lidar_props[self.output]['cmap'])



        super(Lidar, self).__init__(data=Z, x=X, y=Y, **kwargs )
        self.generate()

        print kwargs

        if self.scan  in ['FixedPoint']:
            self.axes[0].xaxis.set_major_formatter(
                DateFormatter('%H:%M \n%d-%b') )

        if 'add_text' in kwargs.keys():
            self.axes[0].text(
                1.01,0.,
                kwargs['add_text'],
                ha='left',
                va='bottom',
                transform=self.axes[0].transAxes )

        self._save_fig(**kwargs)


    def profiler(self,df, **kwargs):
        """Method for building scanning profiles with variations at Zenith or Azimuth

        Parameters
            df          = DataFrame object with Height as columns and Degree variations as index
            """

        profile   = df.copy()

        # print profile.index

        x       = np.empty((profile.shape[0]+1,profile.shape[1]+1))
        y       = x.copy()
        dh      = np.nanmin(profile.index.values[1:] - profile.index.values[:-1])
        dg      = np.nanmin(profile.columns.values[1:] - profile.columns.values[:-1])

        # for ix in range():

        for ix, angle in enumerate(profile.index.values):
            print "{}\n {} = {}".format('='*50,self.degree_variable,angle)
            x[ix,:-1]    = (profile.columns.values - dh/2.) * np.cos(
                                    (angle - dg/2.) * np.pi/180.)
            x[ix,-1]     = (profile.columns.values[-1] + dh/2.) * np.cos(
                                    (angle - dg/2.) * np.pi/180.)
            y[ix,:-1]    = (profile.columns.values - dh/2.) * np.sin(
                                    (angle - dg/2.) * np.pi/180.)
            y[ix,-1]     = (profile.columns.values[-1] + dh/2.) * np.sin(
                                    (angle - dg/2.) * np.pi/180.)

        y[-1,:-1]    = (profile.columns.values - dh/2.) * np.sin(
                                    (angle + dg/2.) * np.pi/180.)
        y[-1,-1]     = (profile.columns.values[-1] + dh/2.) * np.sin(
                                    (angle + dg/2.) * np.pi/180.)
        x[-1,:-1]    = (profile.columns.values - dh/2.) * np.cos(
                                    (angle + dg/2.) * np.pi/180.)
        x[-1,-1]     = (profile.columns.values[-1] + dh/2.) * np.cos(
                                    (angle + dg/2.) * np.pi/180.)


        self.plot_lidar(x, y, profile, **kwargs )

    # def fixedpoint(self, df,**kwargs):
    #     """Method for building fixed point profiles with variations at Zenith or Azimuth
    #
    #     Parameters
    #         Same as plot method
    #     """
    #
    #     if 'df' in kwargs.keys():
    #         data, data_info    = df['data'], df['data_info']
    #     else:
    #         data, data_info    = self.datos.copy(), self.datos_info.copy()
    #
    #     for azi in  data_info[self.degree_variable].drop_duplicates().values:
    #         print "{}\n {} = {}".format('='*50,self.degree_variable,azi)
    #         idi = data_info[data_info[self.degree_variable]!=azi].index
    #
    #         profile             = data[data.index < height].xs(parameter, axis=1,level=1)
    #
    #         profile.loc[:,idi]  = np.NaN
    #
    #         self.plot_lidar(profile.columns.values, profile.index.values, profile, **kwargs)
    #     #
    #     return data_info[angulo_fijo].drop_duplicates().values

    def plot(self, height=4.5, **kwargs):
        """Method for building scanning or fixed point profiles with variations
        at Zenith or Azimuth

        Parameters
            height  = int - For selecting height to plot

            **kwargs
            dates       = DatetimeIndex - Used to plot. Default .index[0]
            colorbar_kind        = Contour choices  - ['Linear', 'Log', 'Anomaly'] - Default Linear
            df          = DataFrame object with Height as index and Degree variations as columns - Allow to use data insted heritance
            parameters  = List like - Choice  parameter to plot - Default use all parameters in self.datos
            output      = Allowed {} """.format(self.lidar_props.keys())

        if 'df' in kwargs.keys():
            self.datos = kwargs.pop('df')

        else:
            self.derived_output(
                output=kwargs.pop('output',None),
                total_signal=kwargs.pop('total_signal',False)
            )
            self.datos = self.datos.loc[
                kwargs.pop('dates',
                    self.datos.index[0]
                    if self.scan != 'FixedPoint' else
                    self.datos.index),
                self.datos.columns.levels[0][
                    self.datos.columns.levels[0] < height ]
            ]
            kwargs['colorbar_kind'] = kwargs.pop(
                        'colorbar_kind',
                        self.lidar_props[ self.output ] ['colorbar_kind'] )



        _vlim = None if 'vlim' in kwargs.keys() else self.get_vlim(**kwargs)

        _path               = kwargs.get('path','')
        kwargs['path']      = "{}{}/{}{}".format(
                                self.plotbook_args['path'],
                                self.scan,
                                _path,
                                '/' if _path[-1] != '/' else ''
                            )
        os.system(
            'ssh {}@siata.gov.co "mkdir /var/www/{}"'.format(
                self.plotbook_args['user'],
                kwargs['path'] )
        )
        os.system('rm Figuras/*')


        if self.scan == 'FixedPoint' :
            self.plot_parameters(_vlim=_vlim, **kwargs)

        else:
            datos = self.datos.copy()
            for date in datos.index:
                self.datos = datos.loc[date]
                kwargs['add_text']  = date.strftime('%b-%d\n%H:%M')
                kwargs['title']     = "{} = {}".format(
                                        self.degree_fixed,
                                        self.degrees_to_cardinal(
                                            self.datos_info.loc[date, self.degree_fixed], self.degree_fixed )
                                    )
                self.plot_parameters(_vlim=_vlim,**kwargs)

        if kwargs.get('make_gif',False) and self.scan != 'FixedPoint':
            gif_kwd = {}
            for col in kwargs.get('parameters',self.datos.columns.levels[-1].values):
                gif_kwd['textsave']     = "_{}_{}_{}".format(
                                                self.scan,
                                                self.output,
                                                col )
                gif_kwd['textsave_gif'] = '{}_{}'.format(
                                                kwargs.get('textsave',''),
                                                kwargs['dates'][0].strftime('%Y-%m-%d') )
                gif_kwd['path']         = kwargs['path']
                self._make_gif(**gif_kwd)

    def plot_parameters(self,_vlim=None,**kwargs):

        for par in  kwargs.pop('parameters',self.datos.columns.levels[-1].values):
            if _vlim is not None:
                kwargs['vlim'] = _vlim[par].values
            self.plot_by_parameter(
                parameter = par,
                **kwargs)



    def plot_by_parameter(self,parameter, **kwargs):

        kwargs['cbarlabel'] = self.lidar_props[self.output][parameter[:6]]
        kwargs['textsave']  = "_{}_{}_{}{}_{}".format(
                                    self.scan,
                                    self.output,
                                    parameter,
                                    kwargs.pop('textsave',''),
                                    self.datos.index[0].strftime(
                                        '%H:%M'
                                        if self.scan != 'FixedPoint' else
                                        '%m-%d' )
                                )

        dataframe           =  self.get_parameter( parameter )

        if self.scan not in ['FixedPoint']:

            self.profiler(
                dataframe,
                **kwargs
            )
        else:
            self.plot_lidar(
                dataframe.columns.values,
                dataframe.index.values,
                dataframe,
                **kwargs
            )

    def get_parameter(self, parameter):
        if self.scan == 'FixedPoint':
            dataframe   = self.datos.xs(
                    (90,parameter),
                    level=[1,2],
                    axis=1 ).T
        else:
            dataframe   = self.datos.xs(
                parameter,
                level=-1,
                ).unstack(0)
        return dataframe

    def get_vlim(self, **kwrgs):
        vlim = self.datos.stack( [0,1] # [0,1] if len(self.datos.columns.names) > 2 else 0
                    ).apply(
                        lambda x: np.nanpercentile(x,[1,99])  )

        if kwrgs.get('colorbar_kind','Linear') == 'Log':
            vlim [ vlim < 0.1 ]     = 0.1
            # vlim                       = np.log10(vlim)
            # vlim [ vlim == -np.inf ]  = 0
            print vlim

        return vlim

    @property
    def Pr(self):

        mvolts  = lambda x:  x * self.datos_info[ 'InputRange_'+x.name[-1] ] * (
            2. ** (-self.datos_info[ 'ADCBits_'+x.name[-1] ])
            ) / self.datos_info[ 'ShotNumber_'+x.name[-1] ] * 1000

        mHz     = lambda x:  x * ( 150 /  self.datos_info['BinWidth_'+x.name[-1] ]
            ) / self.datos_info[ 'ShotNumber_'+x.name[-1] ]

        return self.datos.apply(
            lambda serie:
                mvolts(serie)
                if 'analog' in serie.name[-1] else
                mHz(serie) )

    @property
    def background(self):
        y   = self.datos.loc(axis=1) [
                self.datos.columns.levels[0] [
                    (self.datos.columns.levels[0] > 18) &
                    (self.datos.columns.levels[0] < 21)
                ]
            ].groupby(level=(1,2), axis=1).mean()

        return self.datos.apply(lambda x: x - y[ (x.name[1],x.name[2]) ])
        # self.datos       = self.datos.apply(lambda x: x - self.bkg[x.name[-1]])
        # print self.datos.shape
        # return self.datos.apply(lambda x: x - self.bkg[x.name[-1]])

    @property
    def RCS(self):
        # self.datos       = self.datos.apply(lambda x: x - self.bkg[x.name[-1]])
        # print self.datos.shape
        return self.datos.apply(lambda x: x*( x.name[0]**2) )

    @property
    def total_signal(self):
        an =    self.datos.xs('analog-p',axis=1,level=2
                ) + self.datos.xs('analog-s',axis=1,level=2) * self.V0
        pc =    self.datos.xs('photon-p',axis=1,level=2
                ) + self.datos.xs('photon-s',axis=1,level=2)
        return pd.concat({'analog-b':an,'photon-b':pc}).unstack(0)

    @property
    def depolarization_ratio(self):
        an =    self.datos.xs('analog-p',axis=1,level=2
                ) / self.datos.xs('analog-s',axis=1,level=2)
        pc =    self.datos.xs('photon-p',axis=1,level=2
                ) / self.datos.xs('photon-s',axis=1,level=2)
        return pd.concat({'analog':an,'photon':pc}).unstack(0)

    @property
    def derived(self):
        return self.datos.T.groupby(level=[1,2]).diff().T

    @staticmethod
    def derived2(obj):
        "obj = DataFrame object"
        dr  = obj.columns.levels[0][1] - obj.columns.levels[0][0]
        # df =
        return pd.DataFrame(
                (obj.values[1:] - obj.values[:-1]) / dr,
                index= obj.index[:-1] + dr/2.,
                columns= obj.columns)
        # return df.replace([np.inf, -np.inf], np.NaN)

    @property
    def average_filter(self):
        return  self.datos.rolling(
                    window=30,
                    center=True,
                    min_periods=1,
                    axis=1
                ).median()

    @staticmethod
    def average_filter2(obj,window=30):
        "obj = DataFrame object"
        return  obj.rolling(
                    window,
                    center=True,
                    min_periods=1
                ).median()
        # return obj.groupby()

    @staticmethod
    def degrees_to_cardinal(d,degree='Azimuth'):
        '''
        note: this is highly approximate...
        '''
        # azimuths = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        #         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        dirs = ["E", "ENE", "NE", "NNE",
                "N", "NNW", "NW", "WNW",
                "W", "WSW", "SW", "SSW",
                "S", "SSE", "SE","ESE"]
        ix = int((d + 11.25)/22.5)

        if degree == 'Azimuth':
            return dirs[ix % 16]
        else:
            return d

# def ceilometro(Fecha_Inicio, Fecha_Fin,ceilometro='amva'): #'siata', 'itagui'
#     locale.setlocale(locale.LC_TIME, ('en_us','utf-8'))
#
#     Fecha_1 = datetime.strptime(Fecha_Inicio,'%Y-%m-%d %H:%M:%S') + timedelta(hours=5)
#     Fecha_2 = datetime.strptime(Fecha_Fin,'%Y-%m-%d %H:%M:%S') + timedelta(hours=5) + timedelta(days=1)
#
#     File_List = pd.date_range(Fecha_1, Fecha_2, freq='1D')
#
#     Backs  = pd.DataFrame()
#     Fechas = []
#
#     for idd, Fecha in enumerate(File_List):
#        fname  = Fecha.strftime( 'jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro{}/%Y/%b/CEILOMETER_1_LEVEL_2_%d.his'.format(ceilometro))
#
#        os.system ("scp {} Datos/".format(fname))
#        try:
#
#             Backs = Backs.append( pd.read_csv( fname[-27:], usecols=[0,4], index_col=0, header=1, parse_dates=True, names=['Fecha','Bks'], converters = {'Bks':decode_hex_string} )  )
#             BIN_fname  = np.genfromtxt('Datos/%s' %fname[-27:],delimiter=', ',dtype=object,usecols=(0,4),skip_header=2,\
#                        converters = {0: lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")})
#
#
#             DATA = np.array([decode_hex_string(BIN_fname[i,1]) for i in range(len(BIN_fname))]).T
#
#             File_Dates = np.array(BIN_fname[:,0].tolist())
#
#         except:
#             continue
#
#         if idd == 0:
#             Backs  = DATA
#             Fechas = File_Dates
#         else:
#             Backs = np.concatenate([Backs,DATA],axis=1)
#             Fechas = np.concatenate([Fechas, File_Dates])
#
#         os.system('rm Datos/{}'.format(fname[-27:]))
#
#     Backs = promedio(Backs, 3, 15)
#     Backs = Backs.astype(np.float)
#     Backs[Backs < 0] = np.NaN
#
#     Backs = pd.DataFrame(Backs.T, np.array(Fechas) - timedelta(hours=5))
#     Backs = Backs[Fecha_Inicio:Fecha_Fin]
#     locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))
#     return Backs

# def twos_comp(val, bits):
#     if((val & (1 << (bits - 1))) != 0):
#         # print True
#         val = val - (1 << bits)
#     return val
#
# def decode_hex_string(string, fail_value=1, char_count=5, use_filter=False):
#     data_len    = len(string)
#     print '\nSize string = {}\n'.format(data_len)
#     data        = np.zeros(data_len / char_count, dtype=int)
#     key         = 0
#     for i in xrange(0, data_len, char_count):
#         hex_string      = string[i:i + char_count]
#         data[key]       = twos_comp(int(hex_string, 16), 20)
#         key             += 1
#     if use_filter:
#         data[data <= 0] = fail_value
#         data            = np.log10(data) - 9.
#     # print data[0]
#     return data


#
# Fecha = pd.to_datetime('2018-08-03')
# ceilometro = 'amva'
# fname  = Fecha.strftime( 'jhernandezv@192.168.1.62:/mnt/ALMACENAMIENTO/ceilometro/datos/ceilometro{}/%Y/%b/CEILOMETER_1_LEVEL_2_%d.his'.format(ceilometro))
# x =  pd.read_csv( fname[-27:], usecols=[0,4], index_col=0, header=1, parse_dates=True, names=['Fecha','Bks'], delimiter=', ' ) #converters = {'Bks':decode_hex_string}
# # x3.apply(lambda x: x.str.len())
# # x.applymap(decode_hex_string)
#
# height  = np.arange(0.01,4.51,0.01)
# bks = pd.DataFrame()
# np.empty(x.size)
# for ix in x.index:
#     print ix
#     bks = bks.append(pd.DataFrame(decode_hex_string(x.loc[ix,'Bks']),index=height,columns=[ix]))
# # ix = x.index[0]
# pd.DataFrame(decode_hex_string(x.loc[ix,'Bks']),index=height,columns=[ix])
