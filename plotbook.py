# -*- coding: utf-8 -*-
from matplotlib import use
use('PDF')

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting._core import MPLPlot


plt.rc(	'font',
	size = 18,
	family = FontProperties(
		fname = '/home/jhernandezv/Tools/AvenirLTStd-Book.ttf'
		).get_name()
)

typColor = '#%02x%02x%02x' % (115,115,115)
plt.rc('axes',labelcolor=typColor, edgecolor=typColor,)#facecolor=typColor)
plt.rc('axes.spines',right=False, top=False, )#left=False, bottom=False)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)
plt.rc('figure.subplot', left=0, right=1, bottom=0, top=1)




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

class PlotBook(MPLPlot):

    """Subclass for ploting easily by inheritance of pandas.plottin.core.MPLPlot

    **args
        ax          = Matplotlib axes object, optional
        textsave    = string - file name saved.
        formato      = string - file format type saved. default 'png'
        title       = string - title plot.
        path        = link to save files in /var/www/. default 'jhernandezv/Lidar/'
        add_text    = string - add text.
        scp         = bolean - copy to path. default True
        colorbar_kind    = contour choices  - Linear, Log, Anomaly
        kwds        = matplotlib kwargs allowed
    """

    os.system('mkdir Figuras')
    plotbook_args         = {
        'formato': 'png',
        'local_path':'Figuras/Lidar',
        'path': 'jhernandezv/Lidar/',
        'scp':True,
        'textsave': '',
        'user':'jhernandezv',
        'delay':30,
        'textsave_gif':'',
        'colorbar_kind': 'Linear',
        'ylabel': '',
        'xlabel':'',
        'cbarlabel':'',
        'format': '%.f',
        'vlim': None,

    }

    def __init__(self, data, x, y, *args, **kwargs):

        self.kwargs = self.plotbook_args.copy()

        for kw in self.plotbook_args.keys():
            if kw in kwargs.keys():
                self.kwargs[kw] = kwargs.pop(kw)

        super(PlotBook, self).__init__(data,*args,**kwargs)
        self.x = x
        self.y = y


    def _save_fig(self,**kwargs):

        # print 'Kwargs PlotBook.method \n {}'.format(kwargs)
        # self.kwargs.update(kwargs)
        kwg = self.plotbook_args.copy()
        kwg.update(kwargs)
        plt.savefig(
			'{local_path}{textsave}.{formato}'.format(**kwg) ,
			bbox_inches="tight"
		)
        if kwg['scp']:
            os.system('scp "{local_path}{textsave}.{formato}" {user}@siata.gov.co:/var/www/{path}'. format(**kwg) )
            os.system('scp "{local_path}{textsave}.{formato}" {user}@siata.gov.co:/var/www/{path}'. format(**kwg) )
        plt.close('all')

    def _make_gif(self,**kwargs):

        kwg = self.plotbook_args.copy()
        kwg.update(kwargs)
        os.system( 'convert -delay {delay} -loop 0 "{local_path}{textsave}*" "{local_path}{textsave}{textsave_gif}.gif"'.format(**kwg))
        os.system('scp "{local_path}{textsave}{textsave_gif}.gif" {user}@siata.gov.co:/var/www/{path}'.format(**kwg) )
        os.system('scp "{local_path}{textsave}{textsave_gif}.gif" {user}@siata.gov.co:/var/www/{path}'.format(**kwg) )

    def _make_contour(self, **kwargs):

        """ method for generalizanting contour and colorbar plotting

            kwargs
            cax  = matplotlib.Axes used to print colorbar. Default use make_axes_locatable

        """

        Z = self.data.copy()

        if self.kwargs['vlim'] is not None:
            vmin, vmax      = self.kwargs['vlim']
        else:
            vmin, vmax      = [Z.min().min(),Z.max().max()] #np.nanpercentile(Z,[2.5,97.5]) #
        print vmin, vmax
        colorbar_kwd        = {'extend':'both'}
        contour_kwd         = { 'levels':np.linspace(vmin,vmax,100),
                                'cmap':self.colormap }

        if self.kwargs['colorbar_kind'] == 'Linear':
            contour_kwd['norm']    = Normalize(vmin,vmax)

        elif self.kwargs['colorbar_kind'] == 'Anomaly':
            contour_kwd['norm']     = MidpointNormalize(
										midpoint=0.,
										vmin=vmin,
										vmax=vmax)
            colorbar_kwd['format']  = self.kwargs['format']

        elif self.kwargs['colorbar_kind'] == 'Log':
            Z.mask( Z <= 0, inplace=True )
            if  self.kwargs['vlim'] is None:
                vmin, vmax =  [Z.min().min(),Z.max().max()]  # np.nanpercentile(Z,[1,99])))

            vmin, vmax =   np.log10([vmin,vmax])

            contour_kwd['levels']               = np.logspace(vmin,vmax,100)
            Z[Z < contour_kwd['levels'][0]]     = contour_kwd['levels'][0]
            Z[Z > contour_kwd['levels'][-1]]    = contour_kwd['levels'][-1]
            contour_kwd['norm']                 = LogNorm(
                                        contour_kwd['levels'][0],
										contour_kwd['levels'][-1] )
            print contour_kwd['levels'][0],contour_kwd['levels'][-1]
            # print Z

            minorticks = np.hstack([np.arange(1,10,1)*log for log in np.logspace(-2,16,19)])
            minorticks = minorticks[(minorticks >=contour_kwd['levels'] [0]) & (minorticks <=contour_kwd['levels'] [-1])]
            colorbar_kwd.update(dict(format = LogFormatterMathtext(10) ,ticks=LogLocator(10) ))

        # cf		= ax.contourf(X,Y,Z,levels=levels, alpha=1,cmap =shrunk_cmap,  norm=LogNorm())   #
        contour_kwd.pop('levels')
        args    = (self.x, self.y, Z)
        cf      = self.axes[0].pcolormesh(*args, **contour_kwd)

        # cf		= ax.contourf(X,Y,Z,**contour_kwd) #extend='both')
        if 'cax' not in kwargs.keys():
            divider 	= make_axes_locatable(self.axes[0])
            cax         = divider.append_axes("right", size="3%", pad='1%')
        else:
            cax = kwargs['cax']

        cbar     = plt.colorbar(cf, cax = cax , **colorbar_kwd)
        cbar.set_label(self.kwargs['cbarlabel'])

        if self.kwargs['colorbar_kind']  == 'Log':
            cbar.ax.yaxis.set_ticks(cf.norm(minorticks), minor=True)
            cbar.ax.tick_params(which='minor',width=1,length=4)
            cbar.ax.tick_params(which='major',width=1,length=6)
            # ax.yaxis.set_minor_locator(LogLocator(10,subs=np.arange(2,10)))

        self.kwargs['colorbar_kind'] = 'Linear'
        return cf, cbar

    # def _make_plot(self):
    #     print 'Overwrite method in child Classes'
    #     pass

    def _post_plot_logic(self, ax, data):
        ax.set_ylabel(self.kwargs['ylabel'])
        ax.set_xlabel(self.kwargs['xlabel'])
