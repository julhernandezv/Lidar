#!/usr/bin/env python

from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *


# from pyhdf import SD
from pyhdf.HDF import *
from pyhdf.V   import *
from pyhdf.VS  import *
from pyhdf.SD  import *

# from numpy import *         # hay que importar todo para que funcione pyhdf
import numpy as np
import os, locale, sys, zipfile
# from netCDF4 import Dataset
import datetime as dt
import matplotlib
matplotlib.use("template")

import pprint
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from mpl_toolkits.basemap import Basemap

#----------------------------------------------------------------------------------------#
file_path = '/mnt/Sanduchera/ftp1.cloudsat.cira.colostate.edu/2B-CLDCLASS-LIDAR.P_R04/'
year = '2008'
day  = '183'

# file_path  = ''
file_name = '2008183012329_11573_CS_2B-CLDCLASS-LIDAR_GRANULE_P_R04_E02.hdf'
path_fig  = '/home/cmcuervol/A-Train/CloudSat/Figures/'
#----------------------------------------------------------------------------------------#

Path_fuentes = '/home/cmcuervol/Fuentes/' # Atlas
# Path_fuentes = '/home/ccuervo/Fuentes/' # SAL


# Colors for graphics SIATA style
gris70 = (112/255., 111/255., 111/255.)

ColorInfo1 = (82 /255., 183/255.,196/255.)
ColorInfo2 = (55 /255., 123/255.,148/255.)
ColorInfo3 = (43 /255.,  72/255.,105/255.)
ColorInfo4 = (32 /255.,  34/255., 72/255.)
ColorInfo5 = (34 /255.,  71/255., 94/255.)
ColorInfo6 = (31 /255., 115/255.,116/255.)
ColorInfo7 = (39 /255., 165/255.,132/255.)
ColorInfo8 = (139/255., 187/255.,116/255.)
ColorInfo9 = (200/255., 209/255., 93/255.)
ColorInfo10 =(249/255., 230/255., 57/255.)


AzulChimba =( 55/255., 132/255., 251/255.)
AzulChimbita =( 16/255., 108/255., 214/255.)
VerdeChimba =(  9/255., 210/255.,  97/255.)
Azul =( 96/255., 200/255., 247/255.)
Naranja =( 240/255., 108/255.,  34/255.)
RojoChimba =( 240/255., 84/255.,  107/255.)
Verdecillo =( 40/255., 225/255.,  200/255.)
Azulillo =( 55/255., 150/255.,  220/255.)
# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


# Types of fonts Avenir
#
AvenirHeavy = fm.FontProperties(fname=Path_fuentes+'AvenirLTStd-Heavy.otf')
AvenirBook  = fm.FontProperties(fname=Path_fuentes+'AvenirLTStd-Book.otf')
AvenirBlack = fm.FontProperties(fname=Path_fuentes+'AvenirLTStd-Black.otf')
AvenirRoman = fm.FontProperties(fname=Path_fuentes+'AvenirLTStd-Roman.ttf')

def Listador(directorio, inicio=None, final=None):
    """
    Return the elements (files and directories) of any directory,
    optionaly with filter by start o end of the name of the element
    IMPUTS
    directorio : route of the directory
    inicio     : start of the elements
    final      : end of the elements
    OUTPUTS
    lf  : list of elements
    """
    lf = []
    lista = os.listdir(directorio)
    lista.sort()
    if inicio == final == None:
        return lista
    else:
        for i in lista:
            if inicio == None:
                if i.endswith(final):
                    lf.append(i)
            if final == None:
                if i.startswith(inicio):
                    lf.append(i)
        return lf


def HDFvars(File):
    """
    Extrae los nombres de las variables para un archivo (File)
    """
    # hdfFile = SD.SD(File, mode=1)
    hdfFile = SD(File, mode=1)
    dsets = hdfFile.datasets()
    k = []
    for key in dsets.keys():
        # print key
        k.append(key)
    k.sort()
    hdfFile.end()
    return k



def DesHDF(File, name_var, forma=''):
    """
    Extrae los datos de HDF para un archivo(file), y para una varaiable (name_var),
    puede ser Ascendente ('_A') o descendente ('_D')
    """
    # hdfFile = SD.SD(File, mode=1)
    hdfFile = SD(File, mode=1)
    # print 'Reading ', File
    d1 = hdfFile.select(name_var+forma)
    var = d1.get()
    print 'Extracting ', name_var
    d1.endaccess()
    hdfFile.end()

    return var


def HDFread(filename, variable, Class=None):
    """
    Extract the data for non scientific data in V mode of hdf file
    """
    hdf = HDF(filename, HC.READ)

    # Initialize the SD, V and VS interfaces on the file.
    sd = SD(filename)
    vs = hdf.vstart()
    v  = hdf.vgstart()

    # Encontrar el puto id de las Geolocation Fields
    if Class == None:
        ref = v.findclass('SWATH Vgroup')
    else:
        ref = v.findclass(Class)

    # Open all data of the class
    vg = v.attach(ref)
    # All fields in the class
    members = vg.tagrefs()

    nrecs = []
    names = []
    for tag, ref in members:
        # Vdata tag
        vd = vs.attach(ref)
        # nrecs, intmode, fields, size, name = vd.inquire()
        nrecs.append(vd.inquire()[0]) # number of records of the Vdata
        names.append(vd.inquire()[-1])# name of the Vdata
        vd.detach()

    idx = names.index(variable)
    var = vs.attach(members[idx][1])
    V   = var.read(nrecs[idx])
    var.detach()
    # Terminate V, VS and SD interfaces.
    v.end()
    vs.end()
    sd.end()
    # Close HDF file.
    hdf.close()

    return np.array(V).ravel()

def Salto(x, delta):
    "Determina cuando hay un cambio en un array"
    a = np.array(map(lambda i: x[i+1]-x[i], range(len(x)-1)))
    try:
        pos = np.where(a!=delta)[0][0]
    except:
        pos = 0
    return pos

def Unzipeador(path, ext='.zip', erase=True, path_extract=None):
    """
    Extract and remove the zip files of a directory
    IMPUTS
    path : directory with the zip files
    ext  : files extention
    erase: True to erase the zip files after extract
    path_extract : directory to extract the zip files,
                   if is None, the files are extract in the same directory (path)
    """
    files = Listador(path, final=ext)
    for File in files:
        zip_ref = zipfile.ZipFile(path+File)
        if path_extract is None:
            zip_ref.extractall(path)
        else:
            zip_ref.extractall(path_extract)
        print File
        zip_ref.close()
        if erase == True:
            os.remove(path+File)

def GranuleTime(Filename):
    """
    Convert to datetime format the date of granule data for CloudSat
    """
    Year = int(Filename[ :  4])
    JyDy = int(Filename[ 4: 7])
    Hour = int(Filename[ 7: 9])
    Min  = int(Filename[ 9:11])
    Seg  = int(Filename[11:13])
    GranuTime = dt.datetime(Year,1,1,Hour,Min,Seg)+dt.timedelta(days=JyDy-1)
    return GranuTime






#----------------------------------------------------------------------------------------#
Unzipeador(file_path+year+'/'+day+'/', erase=False)
fecha = GranuleTime(file_name)
# Read HDF Files (VD data) Latitude & Longitude

f = HDF(file_path+year+'/'+day+'/'+file_name, SDC.READ)
vs = f.vstart()

Latitude = vs.attach('Latitude')
Longitude = vs.attach('Longitude')

Lat = np.array(Latitude[:]).ravel()
Lon = np.array(Longitude[:]).ravel()

Latitude.detach()
Longitude.detach()
vs.end()
f.close()

#----------------------------------------------------------------------------------------#
# SDS Data
Vars = HDFvars(file_path+year+'/'+day+'/'+file_name)

file = SD(file_path+year+'/'+day+'/'+file_name, SDC.READ)

file_info = file.info()
print(file_info)  # number of sds and metadata

def DescribeHDFvar(filename, variable):
    """
    Describes the info and atributes of HDF variable
    IMPUTS
    filename : complete path and filename
    variable : name of the variable to describe, type STR
    """

    file = SD(filename, SDC.READ)
    print('---------- '+variable+ ' ----------')
    sds_obj = file.select(variable)
    Var = sds_obj.get()
    sds_info = sds_obj.info()
    print(Var.shape)
    print( sds_info )
    print( sds_info[0], sds_info[1] )
    print( 'sds attributes' )
    pprint.pprint( sds_obj.attributes() )
    file.end()


DescribeHDFvar(file_path+year+'/'+day+'/'+file_name,'CloudLayerBase')
DescribeHDFvar(file_path+year+'/'+day+'/'+file_name,'CloudLayerTop')
DescribeHDFvar(file_path+year+'/'+day+'/'+file_name,'CloudPhase')

CloudLayerBase = DesHDF(file_path+year+'/'+day+'/'+file_name,'CloudLayerBase')
CloudLayerTop  = DesHDF(file_path+year+'/'+day+'/'+file_name,'CloudLayerTop')
CloudPhase     = DesHDF(file_path+year+'/'+day+'/'+file_name,'CloudPhase')


cldclass_lidar_start_idx = 000
# cldclass_lidar_end_idx = 1000
cldclass_lidar_end_idx = CloudPhase.shape[0]

tag = '2'

#----------------------------------------------------------------------------------------#
# Plot cldclass-lidar cloud layers + nb cloud phase
def GranulePloterClass(Base, Top, Phase, lats, lons,
                       fecha='15-01-1992', name='Prueba'):
    """
    Plot a CloudSat data "Granule" is defined as one orbit. A granule starts
    at the first profile that falls on or past the equator on the descending node.
    (note: Approximately 20 seconds, or 125 profiles, will be appended to the
    beginning and end of each granule).

    IMPUTS:
    Base  : Cloud layer base [km]
    Top   : Cloud layer top [km]
    Phase : Cloud phase; 0: Nocloud, 1: ICE, 2: Mixed, 3:Liquid
    lats  : array of latitudes
    lons  : array of longitudes
    fecha : datetime of the data
    name  : name of the outputfile

    OUTPUTS
    file save in the path_fig folder
    """

    def OrganizaLon(longitude):
        """
        reacomodate the longitudes values of a satellite track for graph it easily
        """
        n   = np.where(longitude>0)[0] # Indixes with longitudes positives
        a   = Salto(n,1)         # When occurs the change to negative
        idx = n[a+1]             # Index when is positive again
        copi = longitude.copy()
        copi[idx:] = copi[idx:]-360 # Change to negative te last positive portion of data

        return copi

    #limpiar guevonadas
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1)

    lons = OrganizaLon(lons)+360
    m = Basemap(ax=ax1,llcrnrlat=-90, llcrnrlon=lons.min(),
                urcrnrlat=90, urcrnrlon=lons.max())
    m.shadedrelief()
    # draw parallels.
    parallels = np.arange(-90.,91.,30.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=7)
    # draw meridians
    meridians = np.arange(-360.,721.,60.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=7)
    ax1.set_title('CloudSat track',fontproperties=AvenirRoman, color=gris70, fontsize=15)
    ax1.set_title(fecha.strftime('%Y-%m-%d %H:%M:%S'), loc='right', fontsize=10, color=gris70)
    x, y = m(lons, lats)
    m.plot(x, y,'--', color=RojoChimba,lw=2)

    ax2 = fig.add_subplot(2,1,2)
    for i in range(cldclass_lidar_start_idx,cldclass_lidar_end_idx):
        nb_cloud_layer = np.where(Base[i,:] < 0 )[0][0]
        print('%s of %s' %(i,CloudPhase.shape[0]))
        for j in range(nb_cloud_layer):
            if Base[i,j] > 0 and Top[i,j] > 0.0:
                bar_xi = i
                bar_width = 1.7 # The CloudSat data footprint is approximately 1.7km along-track by 1.3km across-track.
                bar_base = Base[i,j]
                bar_height = Top[i,j] - bar_base
                bar_color = AzulChimba
                bar_edgecolor = AzulChimba
                if Phase[i,j] == 3:
                    bar_color = 'blue'
                    bar_edgecolor = 'blue'
                if Phase[i,j] == 2:
                    bar_color = 'salmon'
                    bar_edgecolor = 'salmon'
                ax2.bar(left=bar_xi, height=bar_height, width=bar_width, bottom=bar_base, color=bar_color, edgecolor=bar_edgecolor)

    # plt.set_facecolor('xkcd:lightblue')
    # ax2.set_facecolor('xkcd:lightblue')

    ax2.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color=AzulChimba, edgecolor=AzulChimba, label='ice clouds')
    ax2.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color='salmon', edgecolor='salmon', label='mixed clouds')
    ax2.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color='blue', edgecolor='blue', label='liquid clouds')

    ax2.legend()

    ax2.set_xlim(cldclass_lidar_start_idx,cldclass_lidar_end_idx)
    ax2.set_ylim(0,20)

    # plt.title('CloudClass-lidar')

    ax2.set_ylabel('Ray-path[km]',fontproperties=AvenirRoman, color=gris70, fontsize=15)
    ax2.set_xlabel('Distance along-track [km]',fontproperties=AvenirRoman, color=gris70, fontsize=15)

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.8, top=0.95, wspace=0.2, hspace=0.1)
    plt.savefig(path_fig + name, transparent=True)


GranulePloterClass(CloudLayerBase,CloudLayerTop,CloudPhase, Lat, Lon,
                   fecha,'CloudClassPhase'+fecha.strftime('%Y-%m-%d_%H-%M-%S')+'.png')

#
# f = plt.figure()
# ax = f.add_subplot(111)
#
# #for i in range(CloudLayerBase.shape[0]):
# for i in range(cldclass_lidar_start_idx,cldclass_lidar_end_idx):
#     nb_cloud_layer = np.where(CloudLayerBase[i,:] < 0 )[0][0]
#     print('%s of %s' %(i,CloudPhase.shape[0]))
#     for j in range(nb_cloud_layer):
#         if CloudLayerBase[i,j] > 0 and CloudLayerTop[i,j] > 0.0:
#             bar_xi = i
#             bar_width = 1.7 # The CloudSat data footprint is approximately 1.7km along-track by 1.3km across-track.
#             bar_base = CloudLayerBase[i,j]
#             bar_height = CloudLayerTop[i,j] - bar_base
#             bar_color = AzulChimba
#             bar_edgecolor = AzulChimba
#             if CloudPhase[i,j] == 3:
#                 bar_color = 'blue'
#                 bar_edgecolor = 'blue'
#             if CloudPhase[i,j] == 2:
#                 bar_color = 'salmon'
#                 bar_edgecolor = 'salmon'
#             plt.bar(left=bar_xi, height=bar_height, width=bar_width, bottom=bar_base, color=bar_color, edgecolor=bar_edgecolor)
#
# ax.set_facecolor('xkcd:lightblue')
#
#
# plt.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color=AzulChimba, edgecolor=AzulChimba, label='ice clouds')
# plt.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color='salmon', edgecolor='salmon', label='mixed clouds')
# plt.bar(left=cldclass_lidar_start_idx, height=0, width=0, bottom=0, color='blue', edgecolor='blue', label='liquid clouds')
#
# plt.legend()
#
# plt.xlim(cldclass_lidar_start_idx,cldclass_lidar_end_idx)
# plt.ylim(0,20)
#
# # plt.title('CloudClass-lidar')
# plt.ylabel('Ray-path[km]')
# plt.xlabel('Distance along-track [km]')
#
# plt.savefig(path_fig+ 'CloudPhase'+fecha.strftime('%Y-%m-%d_%H-%M-%S')+'.png', dpi=100, bbox_inches='tight' )
# #plt.show()
#
# # plt.close()
#
# print ("Hello world")
