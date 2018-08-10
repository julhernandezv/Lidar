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
typColor = '#%02x%02x%02x' % (115,115,115)
plt.rc('axes',labelcolor=typColor,edgecolor=typColor,)
plt.rc('axes.spines',right=False,top=False,left=True)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)


reload (sys)
sys.setdefaultencoding ("utf-8")
locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))


from pandas.plotting._tools import (_subplots, _flatten, table,
                                    _handle_shared_axes, _get_all_lines,
                                    _get_xlim, _set_ticks_props,
                                    format_date_labels)







#
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    cdict = {'red': [],'green': [],'blue': [],'alpha': []}

    reg_index = np.linspace(start, stop, 257)

    shift_index = np.hstack([

        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)

    plt.register_cmap(cmap=newcmap)
    return newcmap

shrunk_cmap = shiftedColorMap(mpl.cm.jet, start=0.0,midpoint=0.65, stop=0.85, name='shrunk')

class MidpointNormalize(mpl.colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class PlotBook():
    """Subclass for ploting kwargs, properties and decorators

    **Kwargs
        ax          = Matplotlib axes object, optional
        textsave    = string - file name saved.
        format      = string - file format type saved. default 'png'
        title       = string - title plot.
        path        = link to save files in /var/www/. default 'jhernandezv/Lidar/'
        add_text    = string - add text.
        scp         = bolean - copy to path. default True
    """

    os.system('mkdir Figuras')
    kwargs         = {
        'format': 'png',
        'local_path':'Figuras/Lidar',
        'path': 'jhernandezv/Lidar/',
        'scp':True,
        'textsave': '',
        'user':'jhernandezv',
        'delay':30,
        'textsave_gif':'',
    }

    def __init__(self, ax=None,fig=None,subplots=False,figsize=None,**kwargs):


        self.ax             = ax
        self.fig            = fig
        self.subpltos       = subplots

        self.kwargs.update(kwargs)

    def _save_fig(self,**kwargs):
        # print 'Kwargs PlotBook.method \n {}'.format(kwargs)
        # self.kwargs.update(kwargs)
        kwg = self.kwargs.copy()
        kwg.update(kwargs)
        plt.savefig('{local_path}{textsave}.{format}'.format(**kwg) ,bbox_inches="tight")
        if kwg['scp']:
            os.system('scp "{local_path}{textsave}.{format}" {user}@siata.gov.co:/var/www/{path}'. format(**kwg) )

    def _make_gif(self,**kwargs):
        kwg = self.kwargs.copy()
        kwg.update(kwargs)
        os.system( 'convert -delay {delay} -loop 0 "{local_path}{textsave}*" "{local_path}{textsave}{textsave_gif}.gif"'.format(**kwg))
        os.system('scp "{local_path}{textsave}{textsave_gif}.gif" {user}@siata.gov.co:/var/www/{path}'.format(**kwg) )

    @property
    def nseries(self):
        if self.data.ndim == 1:
            return 1
        else:
            return self.data.shape[1]

    def generate(self):
        self._args_adjust()
        self._compute_plot_data()
        self._setup_subplots()
        self._make_plot()
        self._add_table()
        self._make_legend()
        self._adorn_subplots()

        for ax in self.axes:
            self._post_plot_logic_common(ax, self.data)
            self._post_plot_logic(ax, self.data)

    def _setup_subplots(self):
        if self.subplots:
            fig, axes = _subplots(naxes=self.nseries,
                                  sharex=self.sharex, sharey=self.sharey,
                                  figsize=self.figsize, ax=self.ax,
                                  layout=self.layout,
                                  layout_type=self._layout_type)
        else:
            if self.ax is None:
                fig = self.plt.figure(figsize=self.figsize)
                axes = fig.add_subplot(111)
            else:
                fig = self.ax.get_figure()
                if self.figsize is not None:
                    fig.set_size_inches(self.figsize)
                axes = self.ax

        axes = _flatten(axes)

        if self.logx or self.loglog:
            [a.set_xscale('log') for a in axes]
        if self.logy or self.loglog:
            [a.set_yscale('log') for a in axes]

        self.fig = fig
        self.axes = axes


class Lidar(PlotBook):
    """
    Class for manipulating SIATA's Scanning Lidar

    Parameters
        output      = 'raw_data','P(r)', 'RCS','Ln(RCS)','fLn(RCS)','dLn(RCS)','fdLn(RCS)','dfLn(RCS)','fdfLn(RCS)'... - options for derived outputs
        scan        = ['Zenith','Azimuth','3D','FixedPoint'] - kind of spacial measurement
        ascii       = bolean - if False read binary files

    """

    # mpl.cm.gist_earth_r}
    label   = {'raw_data':{'analog':'','photon':'','cmap':shrunk_cmap},
                'P(r)':{'analog':r'$[mV]$','photon':r'$[MHz]$','cmap':shrunk_cmap},
                'RCS':{'analog':r'RCS $[mV*Km^2]$','photon':r'RCS $[MHz*Km^2]$','cmap':shrunk_cmap},
                'Ln(RCS)':{'analog':r'Ln(RCS) $[Ln(mV*Km^2)]$','photon':r'Ln(RCS) $[Ln(MHz*Km^2)]$','cmap':shrunk_cmap},
                'fLn(RCS)':{'analog':r'fLn(RCS) $[Ln(mV*Km^2)]$','photon':r'fLn(RCS) $[Ln(MHz*Km^2)]$','cmap':shrunk_cmap}, 'dLn(RCS)':{'analog':r'dLn(RCS)','photon':r'dLn(RCS)','cmap':mpl.cm.seismic},
                'fdLn(RCS)':{'analog':r'fdLn(RCS)','photon':r'fdLn(RCS)','cmap':mpl.cm.seismic},
                'dfLn(RCS)':{'analog':r'dfLn(RCS)','photon':r'dfLn(RCS)','cmap':mpl.cm.seismic},
                'fdfLn(RCS)':{'analog':r'fdfLn(RCS)','photon':r'fdfLn(RCS)','cmap':mpl.cm.seismic},
    }


    def __init__(self, Fechai=None, Fechaf=None, ascii=False, scan='3D', output='P(r)', **kwargs):

        self.ascii      = ascii
        self.scan       = scan
        self.output     = output
        self.Fechai     = (dt.datetime.now()-relativedelta(months=1)).strftime('%Y-%m-')+'01 01:00' if (Fechaf == None) else Fechai
        self.Fechaf     = (pd.to_datetime(self.Fechai)+ relativedelta(months=1)-dt.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M') if (Fechaf == None) else Fechaf #

        self.degree_variable    = 'Zenith' if self.scan in ['FixedPoint','3D'] else self.scan
        self.degree_fixed       = 'Azimuth' if self.scan in ['FixedPoint','3D','Zenith'] else 'Zenith'
        self.kwargs.update(kwargs)
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
        description ['Fecha'] = datetime.datetime.strptime (lineaLocationArray[1] + " " + lineaLocationArray[2], "%d/%m/%Y %H:%M:%S") - dt.timedelta(hours=5)
        description ['Fecha_fin'] = datetime.datetime.strptime (lineaLocationArray[3] + " " + lineaLocationArray[4], "%d/%m/%Y %H:%M:%S") - dt.timedelta(hours=5)
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
            dataset = pd.read_csv(filename,delimiter='\t',header=9 if self.scan not in ['FixedPoint'] else 8,usecols=[0,1,2,3] )
            ejex = np.array(range(1,dataset.shape[0]+ 1))*dictDescripcionDataset[1]["datasetBinWidth"] / 1000.

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
            dataset.index.name  = 'Heigth'
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

        data_info = pd.DataFrame()
        data = {}

        for file in sorted(filenames):
            print "{} \n {} \n {}".format("="*50,file,"="*50)
            df1, df2 = self.read_file(file)
            # print df2.index.strftime('%Y-%m-%d %H:%M:%S')
            # print '\n index= {}'.format(df2['Zenith' if self.scan in ['FixedPoint','3D'] else self.scan].values[0])
            data[ df2.index[0].strftime('%Y-%m-%d %H:%M:%S') ] = df1
            data_info = data_info.append(df2)

        data = pd.concat(data,axis=1)
        data_info = data_info.sort_index() #.reset_index()
        data.columns.set_levels( pd.to_datetime(data.columns.levels[0].values), level=0, inplace=True) #range(data.columns.levels[0].size), level=0, inplace=True)

        if self.scan == '3D':
            if np.abs(data_info.Zenith.iloc[1] - data_info.Zenith.iloc[0]) > 10:
                print data_info.index[0]
                # data_info.loc[data_info.index[0],'Zenith'] = data_info.Zenith.iloc[1] + 5
                data.drop(data_info.index[0],axis=1,level=0,inplace=True)
                data_info.drop(data_info.index[0],inplace=True)

            data_info.loc[data_info.Azimuth != data_info.Azimuth.iloc[0],['Azimuth','Zenith']] += 180
            data_info.loc[data_info.Azimuth == data_info.Azimuth.iloc[0],'Zenith'] *= -1

            #Filter by repeated measures at Zenith 90
            duplicated =  data_info.index[data_info.Zenith == 90]
            print duplicated#duplicated('Zenith',keep=False)

            if duplicated.size >1 :
                print "\n Deleting duplicated"
                data.loc[:, duplicated[0] ] = data[ duplicated ].groupby(axis=1, level=1).mean().values
                data.drop( duplicated[1:], axis=1, level=0, inplace=True)
                data_info.drop( duplicated[1:], inplace=True)

        elif self.scan in ['Zenith','Azimuth','FixedPoint']:
            data_info.loc[:,'Zenith']   *= -1

        data_info['Azimuth']        = (270-data_info['Azimuth'])%360

        data.columns = pd.MultiIndex.from_product(
            [ data_info[ self.degree_variable ].values, data.columns.levels[-1].values ],
            names = [self.degree_variable,'Parameters'] )

        #Filtro para mediciones inferiores a 110m de distacia al sensor
        data       = data[data.index >= 0.110]

        if kwargs.get('inplace',True):
            self.data       = data
            self.data_info  = data_info
            self.raw_data   = self.data.copy()
            self.derived_output(**kwargs)

        else:
            return data, data_info



    def read(self, **kwargs):

        kind_folder = {'3D':'3D','Zenith':'Z','Azimuth':'A','FixedPoint':'RM'}
        ## os.system('rm Figuras/*')
        os.system('mkdir Datos')
        # dates = pd.date_range('20180624','20180717',freq='d')
        dates = pd.date_range(self.Fechai,self.Fechaf,freq='d')
        print dates

        self.data       = {}
        self.data_info  = pd.DataFrame()

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
                    self.data[df4.index[0].strftime('%Y-%m-%d %H:%M:%S')] = df3

                    df4.loc[df4.index[0], 'Fecha_fin'] = df4.index[-1]
                    self.data_info = self.data_info.append(df4.iloc[0])

        self.data           = pd.concat(self.data,axis=1)
        self.data.columns.set_levels( pd.to_datetime(self.data.columns.levels[0]), level=0, inplace=True)
        self.data_info.sort_index(inplace=True)
        self.raw_data   = self.data.copy()
        self.derived_output(**kwargs)


    def derived_output(self,**kwargs):
        """Method for derived values

            **kwargs
            output = Allowed {} """.format(self.label.keys())

        if 'output' in kwargs.keys():
            # if kwargs['output'] not in self.label.keys():
            self.output     = kwargs['output']

        if self.output in self.label.keys():
            self.data       = self.raw_data.copy()

            if self.output not in ['raw_data']:
                self.data       = self.Pr
                if 'background' in kwargs.keys():
                    self.data = self.data.groupby(axis=1,level=0).apply(lambda x: x -  pd.concat([kwargs['background']], axis=1,keys=[x.name]) )

            if self.output in ['RCS','Ln(RCS)','fLn(RCS)','dLn(RCS)','fdLn(RCS)','dfLn(RCS)','fdfLn(RCS)']:
                self.data       = self.RCS

                if self.output not in ['RCS']:
                    # self.data.mask(self.data<=0,inplace=True)
                    self.data[ self.data <=0 ] = 0.01
                    self.data       = np.log(self.data)

                if self.output in ['fLn(RCS)','dfLn(RCS)','fdfLn(RCS)']:
                    self.data       = self.average_filter(self.data)

                if self.output in ['dLn(RCS)','fdLn(RCS)','fdfLn(RCS)','dfLn(RCS)']:
                    self.data       = self.derived(self.data)

                if self.output in ['fdLn(RCS)','fdfLn(RCS)']:
                    self.data       = self.average_filter(self.data)

        else:
            print "Output {} not allowed, check other".format(self.output)

    def plot_lidar(self,X,Y,Z,**kwargs):
        """Function for ploting lidar profiles

        Parameters
            X,Y     = one or two dimensional array for grid plot
            Z       = DataFrame object for contour
            kind    = contour choices  - Linear, Log, Anomaly
            **PlotBook kwargs allowed
        """
        # plt.rc('font', size= kwargs.get('fontsize',16))
        Z  = Z.copy()
        kwargs['kind'] =  kwargs.get('kind','Linear')


        if 'ax' not in kwargs.keys():
            plt.close('all')
            if self.scan not in ['FixedPoint']:

                rel         = (Y.max()-Y.min())/(np.abs(X).max()-X.min())
                figsize = ( 10,10*rel) if rel <=1 else (10*(1./rel),10)
            else: figsize = (10,5.6)

            fig 		= plt.figure(figsize=figsize,facecolor=(.7,.7,.7))
            ax2      	= fig.add_axes((1.02,.2,0.02,0.59))
            ax			= fig.add_axes((0,0.,1,1)) #self.fig.add_subplot(111)


        else:
            ax 		= kwargs['ax']


        ax.patch.set_facecolor((.75,.75,.75))
        divider 	= make_axes_locatable(ax)

        vmin, vmax          = kwargs.pop('vlim',[Z.min().min(),Z.max().max()]) #np.nanpercentile(Z,[2.5,97.5]) #
        print vmin, vmax
        colorbar_kwd        = {}
        contour_kwd         = { 'cmap':self.label[self.output]['cmap'], \
                                'levels':np.linspace(vmin,vmax,100)} #,'extend':'both'}

        if kwargs['kind'] == 'Linear':
            contour_kwd['norm']    = mpl.colors.Normalize(vmin,vmax)

        elif kwargs['kind'] == 'Anomaly':
            contour_kwd['norm']     = MidpointNormalize(midpoint=0.,vmin=vmin, vmax=vmax)
            colorbar_kwd['format']  = '%.f'

        elif kwargs['kind'] == 'Log':
            #
            if 'vlim' in kwargs.keys():
                vmin, vmax                 = kwargs.pop('vlim')
            else:
                Z.mask(Z<=0,inplace=True)
                vmin, vmax =  np.log10( [Z.min().min(),Z.max().max()] ) # np.nanpercentile(Z,[1,99])))
            contour_kwd['levels']               = np.logspace(vmin,vmax,100)
            Z[Z < contour_kwd['levels'][0]]     = contour_kwd['levels'][0]
            Z[Z > contour_kwd['levels'][-1]]    = contour_kwd['levels'][-1]
            contour_kwd['norm']                 = mpl.colors.LogNorm(
                    contour_kwd['levels'][0],contour_kwd['levels'][-1] )
            print contour_kwd['levels'][0],contour_kwd['levels'][-1]
            # print Z

            minorticks = np.hstack([np.arange(1,10,1)*log for log in np.logspace(-2,16,19)])
            minorticks = minorticks[(minorticks >=contour_kwd['levels'] [0]) & (minorticks <=contour_kwd['levels'] [-1])]
            colorbar_kwd.update(dict(format = LogFormatterMathtext(10) ,ticks=LogLocator(10) ))

        # cf		= ax.contourf(X,Y,Z,levels=levels, alpha=1,cmap =shrunk_cmap,  norm=mpl.colors.LogNorm())   #
        contour_kwd.pop('levels')
        cf      = ax.pcolormesh(X,Y,Z,**contour_kwd)

        # cf		= ax.contourf(X,Y,Z,**contour_kwd) #extend='both')


        if 'ax' not in kwargs.keys():
            cbar     = plt.colorbar(cf,cax=divider.append_axes("right", size="3%", pad='1%') if 'ax' in kwargs.keys() else ax2, **colorbar_kwd)
            cbar.set_label(kwargs.get('label',r'$[mVolts]$'))

            if kwargs['kind']  == 'Log':
                cbar.ax.yaxis.set_ticks(cf.norm(minorticks), minor=True)
                cbar.ax.tick_params(which='minor',width=1,length=4)
                cbar.ax.tick_params(which='major',width=1,length=6)
                # ax.yaxis.set_minor_locator(LogLocator(10,subs=np.arange(2,10)))

        ax.set_ylabel(r'Range $[Km]$')
        if self.scan not in ['FixedPoint']:
            ax.set_xlim(-np.abs(X).max(),np.abs(X).max())
            ax.set_xlabel(r'Range $[Km]$',)# fontsize=fontsize)
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M \n%d-%b'))

        if 'add_text' in kwargs.keys():
            ax.text(1.01,0.,kwargs['add_text'],ha='left',va='bottom',transform=ax.transAxes)

        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'],loc='right')

        # print kwargs
        self._save_fig(**kwargs)
        # plt.savefig('Figuras/Lidar{textsave}.{format}'.format(**self.kwargs) %(kwargs.get('textsave',''),kwargs.get('format','png')),bbox_inches="tight")
        # if kwargs.get('scp',True):
        #     os.system("scp Figuras/Lidar{textsave}.{format} jhernandezv@siata.gov.co:/var/www/{path}". format(kwargs.get('textsave',''),kwargs.get('format','png'),kwargs.get('path','jhernandezv/Lidar/') ))


    def profiler(self,df, **kwargs):
        """Method for building scanning profiles with variations at Zenith or Azimuth

        Parameters
            df          = DataFrame object with Heigth as index and Degree variations as columns
            """

        profile   = df.copy()

        # print profile.index

        x = np.empty(profile.shape)
        y = np.empty(profile.shape)

        for ix, angle in enumerate(profile.index.values):
            print "{}\n {} = {}".format('='*50,self.degree_variable,angle)
            x[ix,:] = profile.columns.values* np.cos(angle*np.pi/180.)
            y[ix,:] = profile.columns.values* np.sin(angle*np.pi/180.)

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
    #         data, data_info    = self.data.copy(), self.data_info.copy()
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
        """Method for building scanning or fixed point profiles with variations at Zenith or Azimuth

        Parameters
            height  = int - For selecting heigth to plot

            **kwargs
            dates       = DatetimeIndex - Used to plot. Default .index[0]
            kind        = Contour choices  - ['Linear', 'Log', 'Anomaly'] - Default Linear
            df          = DataFrame object with Heigth as index and Degree variations as columns - Allow to use data insted heritance
            parameters  = List like - Choice  parameter to plot - Default use all parameters in self.data
            output      = Allowed {} """.format(self.label.keys())

        if 'output' in kwargs.keys():
            self.derived_output(**kwargs)

        _parameters         = kwargs.get('parameters',self.data.columns.levels[-1].values)
        _textsave           = kwargs.get('textsave','')
        kwargs['path']      = "{}{}/{}/".format( self.kwargs['path'], self.scan, kwargs.get('path','') )
        os.system('ssh {}@siata.gov.co "mkdir /var/www/{}"'.format( self.kwargs['user'], kwargs['path']  ))
        os.system('rm Figuras/*')

        _dates = kwargs.get('dates', [self.data_info.index[0]] if self.scan != 'FixedPoint' else self.data_info.index)

        _vlim = self.get_vlim(height, **kwargs)
        print _vlim

        for date in _dates if self.scan != 'FixedPoint' else [_dates[0]] :
            print date
            kwargs['title']     = "{} = {}".format(self.degree_fixed ,
            self.degrees_to_cardinal( self.data_info.loc[date, self.degree_fixed], self.degree_fixed))

            for parameter in _parameters:
                kwargs['label']     = self.label[self.output][parameter[:6]]
                kwargs['textsave']  = "_{}_{}_{}{}_{}".format( self.scan,self.output,parameter,_textsave,date.strftime('%H:%M' if self.scan != 'FixedPoint' else '%m-%d') )
                vlim                = kwargs.get('vlim',_vlim[ parameter ].values)
                dataframe           = kwargs.get('df', self.get_from(height,parameter, _dates if len(_dates) > 1 and self.scan == 'FixedPoint' else date))

                if self.scan not in ['FixedPoint']:
                    kwargs['add_text']  = date.strftime('%b-%d\n%H:%M')
                    self.profiler(dataframe, vlim=vlim, **kwargs)
                else:
                    kwargs.pop('title',None)
                    self.plot_lidar(dataframe.columns.values, dataframe.index.values, dataframe, vlim=vlim, **kwargs)

        if kwargs.get('make_gif',False) and self.scan != 'FixedPoint':
            gif_kwargs = {}
            for col in _parameters:
                gif_kwargs['textsave']      = "_{}_{}_{}".format(self.scan,self.output,col)
                gif_kwargs['textsave_gif']  = '{}_{}'.format( _textsave, kwargs['dates'][0].strftime('%Y-%m-%d'))
                gif_kwargs['path']          = kwargs['path']
                self._make_gif(**gif_kwargs)


    def get_from(self, height, parameter, date):
        if self.scan == 'FixedPoint':
            dataframe   = self.data[ self.data.index < height][date] \
                            .xs( (90,parameter),level=[1,2],axis=1)
                            # .resample('30s', axis=1, level=0 ).mean()
        # if len(self.data.columns.names) == 2:
        #     dataframe   = self.data[ self.data.index < height ].xs(parameter,level=-1,axis=1).T
        else:
            dataframe   = self.data[ self.data.index < height].xs( (date,parameter),level=[0,-1],axis=1).T
        return dataframe

    def get_vlim(self, height, **kwrgs):
        vlim = self.data[self.data.index < height] \
                    .stack( [0,1] if len(self.data.columns.names) > 2 else 0) \
                    .apply(lambda x: np.nanpercentile(x,[1,99])) #.quantile([.01,.99])

        if kwrgs.get('kind','Linear') == 'Log':
            vlim [ vlim<0 ]           = 0
            vlim                       = np.log10(vlim)
            vlim [ vlim == -np.inf ]  = 0
            print vlim

        return vlim

    @property
    def Pr(self):
        # def mHz(x):
            # return x *  ( 150 /  self.data_info.loc[x.name[0],x.name[-1]+'_BinWidth']) / self.data_info.loc[x.name[0],x.name[-1]+'_ShotNumber']
        # def mvolts(x):
        #     return x * self.data_info.loc[x.name[0],x.name[-1]+'_InputRange'] * 1000 * (2. ** (-self.data_info.loc[x.name[0],x.name[-1]+'_ADCBits'])) / self.data_info.loc[x.name[0],x.name[-1]+'_ShotNumber']

        mvolts  = lambda x:  x * self.data_info.loc[x.name[0],'InputRange_'+x.name[-1]] * 1000 * (2. ** (-self.data_info.loc[x.name[0],'ADCBits_'+x.name[-1]])) / self.data_info.loc[x.name[0],'ShotNumber_'+x.name[-1]]

        mHz     = lambda x:  x * ( 150 /  self.data_info.loc[x.name[0],'BinWidth_'+x.name[-1]]) / self.data_info.loc[x.name[0],'ShotNumber_'+x.name[-1]]

        return self.data.apply( lambda serie: mvolts(serie) if 'analog' in serie.name[-1] else mHz(serie) )

    @property
    def RCS(self):
        return self.data.apply(lambda x: x*x.index.values**2)

    @staticmethod
    def derived(obj):
        "obj = DataFrame object"
        dr  = obj.index[1] - obj.index[0]
        # df =
        return pd.DataFrame( (obj.values[1:] - obj.values[:-1]) / dr , index= obj.index[:-1] + dr/2., columns= obj.columns)
        # return df.replace([np.inf, -np.inf], np.NaN)

    @staticmethod
    def average_filter(obj,window=30):
        "obj = DataFrame object"
        return obj.rolling(window,center=True,min_periods=1).median()

    @staticmethod
    def degrees_to_cardinal(d,degree='Azimuth'):
        '''
        note: this is highly approximate...
        '''
        # azimuths = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        #         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        dirs = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE","ESE"]
        ix = int((d + 11.25)/22.5)

        if degree == 'Azimuth':
            return dirs[ix % 16]
        else:
            return d
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
# binario.plot(textsave='_test5_log',parameters=['photon-p'],output='RCS',kind='Log')
# binario.plot(textsave='_test_4D',parameters=['photon-p'],output='RCS')
# # dd, di = binario.read_folder(files)
# # '-75.5686', '6.2680'
#
# binario.plot(textsave='_test_1',parameters=['photon-p'])
# binario.plot(textsave='_test_1',parameters=['photon-p'],output='RCS')
# binario.plot(textsave='_log_test_1',output='RCS',parameters=['photon-p'],kind='Log')
# binario.plot(textsave='_test_1',output='Ln(RCS)',parameters=['photon-p'])
# binario.plot(textsave='_test_1',output='dLn(RCS)',parameters=['photon-p'],kind='Anomaly')
# binario.plot(textsave='_test_1',output='fLn(RCS)',parameters=['photon-p'])
# binario.plot(textsave='_test_1',output='fdLn(RCS)',parameters=['photon-p'],kind='Anomaly')
# binario.plot(textsave='_test_1',output='dfLn(RCS)',parameters=['photon-p'],kind='Anomaly')
# binario.plot(textsave='_test_1',output='fdfLn(RCS)',parameters=['photon-p'],kind='Anomaly')

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
#

for date in pd.date_range('2018-08-01','2018-08-08',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
    try:
# date = pd.date_range('2018-07-30','2018-07-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):

        binario = Lidar(Fechai=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='FixedPoint')
        binario.read()
        binario.data = binario.data.stack([1,2]).resample('30s', axis=1, level=0 ).mean().unstack([1,2])
        binario.raw_data = binario.data
        binario.data_info = binario.data_info.resample('30s').mean()

        kwgs = dict( height=4.5,)# background= bkg)
        binario.plot(output = 'P(r)',**kwgs )

        binario.plot( output='RCS', **kwgs )

        binario.plot(textsave='_log', output='RCS',kind='Log',  **kwgs)


        binario.plot( output='Ln(RCS)', **kwgs )
        # #

        # binario.plot(output='dLn(RCS)',kind='Anomaly',  **kwgs)

        binario.plot(output='fLn(RCS)', **kwgs)

        # binario.plot(output='fdLn(RCS)',kind='Anomaly',  **kwgs)

        # binario.plot(output='dfLn(RCS)', kind='Anomaly', **kwgs)

        binario.plot(output='fdfLn(RCS)', kind='Anomaly',  **kwgs)

        kwgs = dict(  height=10,textsave='_10km',path='10km')# background= bkg)

        binario.plot(output = 'P(r)', **kwgs )

        binario.plot( output='RCS', **kwgs )


        binario.plot( output='Ln(RCS)', **kwgs )
        # #

        # binario.plot(output='dLn(RCS)',kind='Anomaly',  **kwgs)

        binario.plot(output='fLn(RCS)', **kwgs)

        # binario.plot(output='fdLn(RCS)',kind='Anomaly',  **kwgs)

        # binario.plot(output='dfLn(RCS)', kind='Anomaly', **kwgs)

        binario.plot(output='fdfLn(RCS)', kind='Anomaly',  **kwgs)

        kwgs.pop('textsave')
        binario.plot(textsave='_log_10km', output='RCS',kind='Log',  **kwgs)


    except:
        pass


# # ################################################################################
# # backgroud = '2018-06-30 19:07'
# bkg = pd.read_csv('Background_test.csv',index_col=0,header=[0,1])
# bkg.columns.set_levels(map(lambda x: int(x),bkg.columns.levels[0].values), level=0,inplace=True)
# bkg = bkg.rolling(30,center=True,min_periods=1).mean()
# ################################################################################
#
# altura = 4.5
# # for date in pd.date_range('2018-06-30','2018-06-30',freq='d'): #'2018-06-27','2018-07-14',freq='d'):
# # try:
# date = pd.date_range('2018-06-30','2018-06-30',freq='d')[0] #'2018-06-27','2018-07-14',freq='d'):
#
# binario = Lidar(Fechai=date.strftime('%Y-%m-%d'),Fechaf=date.strftime('%Y-%m-%d'),scan='3D')
# binario.read()
# backup = [binario.raw_data, binario.data_info]
# # binario.data        = backup[0]
# # binario.raw_data    = backup[0]
# # binario.data_info   = backup[1]
#
#
# kwgs = dict(parameters=['photon-p'], dates=binario.data_info.index, make_gif=True, path= date.strftime('%Y-%m-%d-bkg-nonan'),height=altura, background= bkg)





# binario.plot(scp=False, output='P(r)',**kwgs )
#
# binario.plot(scp=False, output='RCS', **kwgs )
#
# binario.plot(textsave='_log', output='RCS',kind='Log', scp=False, **kwgs)
#
#
# binario.plot(scp=False, output='Ln(RCS)', **kwgs )
# # #
#
# binario.plot(output='dLn(RCS)',kind='Anomaly', scp=False, **kwgs)
#
# binario.plot(scp=False,output='fLn(RCS)', **kwgs)
#
# binario.plot(output='fdLn(RCS)',kind='Anomaly', scp=False, **kwgs)
#
# binario.plot(output='dfLn(RCS)', kind='Anomaly',scp=False, **kwgs)
#
# binario.plot(output='fdfLn(RCS)', kind='Anomaly', scp=False, **kwgs)

    # except:
    #     pass
# binario.plot(textsave='_test', parameters=['photon-p'], output='RCS', kind='Log', dates=binario.data_info.index, make_gif=True, path= '2018-07-04',scp=False)
# binario.plot(textsave='_test', parameters=['photon-p'], output='fdfLn(RCS)', kind='Anomaly', dates=binario.data_info.index, make_gif=True, path= '2018-07-04') #,vlim=[-2,4]
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
#     # ax[ix].scatter(x[col],y.loc[y.index [rezago:rezago-18]  if rezago<18 else y.index [18:],col],label=col,)
#     ax[ix].scatter(x[col],y[col],label=col,)
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
#                 self.plot(textsave = '_{}'.format(self.data_info.index[0].strftime('%H:%M')), path = 'jhernandezv/Lidar/Zenith/{}/'.format(d.strftime('%Y%m%d')),kind='Log')
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
#                 self.plot(zenith=False, kind='Log', \
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
#             Zenith [d.strftime('%Y%m%d')] = self.plot(kind='Log',zenith=False, \
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
#                 self.plot(kind='Log', textsave ='_{}'.format(self.data_info.index[0].strftime('%H:%M')), \
#                         path= '{}3D/{}/'.format(self.kwargs['path'],d.strftime('%Y%m%d')), vlim=[6.7,9])
#             except:
#                 Error.append(folders)
#                 pass
#         # os.system('convert -delay 20 -loop 0 {}*.png {}.gif'.format())
#         for col in self.data.columns.levels[1].values:
#             os.system( 'convert -delay 20 -loop 0 Figuras/Lidar_Scanning_RCS_{}_* Figuras/lidar_Scanning_RCS_{}_{}.gif'.format(col,col,d.strftime('%Y%m%d')))
#             os.system('scp Figuras/lidar_Scanning_RCS_{}_{}.gif jhernandezv@siata.gov.co:/var/www/jhernandezv/Lidar/3D/{}/ '.format(col,d.strftime('%Y%m%d'),d.strftime('%Y%m%d')) )
