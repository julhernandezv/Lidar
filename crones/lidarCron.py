# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
import sys, os, time, locale

from multiprocessing import Pool, Process

from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

baseDir = os.path.abspath(__file__)
print sys.argv
kind = sys.argv[1]
baseDir = os.path.join( baseDir, kind )
os.system( 'mkdir {}'.format(baseDir))
os.chdir ( baseDir )


##Decorator for logfile
# def log(func):
#     def wrapper(*args, **kwargs):
#         func_str    = 'Function : %s\n' %func.__name__
#         args_str    = ', '.join(args) + '\n'
#         kwargs_str  = ', '.join([':'.join([str(j) for j in i]) for i in kwargs.iteritems()]) + '\n'
#         with open('lidarWrapper.log', 'a') as f:
#             f.write(func_str)
#             f.write(args_str)
#             f.write(kwargs_str)
#         return func(*args, **kwargs)
#     return wrapper
#
# @log
def plots (obj,  scan, **kwargs):

    kwgs = dict(
        height=4.5 if scan  != 'Zenith' else 6,
        path= 'Operacional',
        operational=True,
        makeGif= scan != 'FixedPoint'
        # cla=True
    )
    kwgs.update(kwargs)


    obj.plot(output = 'LVD',
        totalSignal=True,
        vlim=[0.25,1],
        **kwgs )

    obj.plot(
        output='RCS',
        parameters=['analog-b'],
        vlim = [.15,20],#[1,20],
        **kwgs
    )

    obj.plot(
        parameters=['analog-s','analog-p'],
        vlim=[.1,16], #[1,16]
        output='RCS',
        **kwgs
    )

    obj.plot(
        parameters=['photon-s','photon-p'],
        vlim=[9,200], #[15,350]
        output='RCS',
        **kwgs
    )

    obj.plot(
        output='RCS',
        parameters=['photon-b'],
        vlim = [10,200],
        **kwgs
    )

    obj.plot(output='fdfLn(RCS)' , **kwgs)


def plotting(element, scan, hora=None, **kwargs):
    if hora != None:
        element.raw = element.raw[(now-dt.timedelta(hours=hora)).strftime('%Y-%m-%d %H:%M'):]
    plots(element, scan, **kwargs)
    if scan != 'FixedPoint':
        plots(element, scan,dates=element.raw.index,scp=False,**kwargs)

def lidar_cron(scan):

    now = dt.datetime.now()

    # for scan in ['FixedPoint','3D','Azimuth','Zenith']:
    # try:
        #48 Horas
        instance =   Lidar(
            fechaI=(now-dt.timedelta(hours=48)).strftime('%Y-%m-%d %H:%M'),
            fechaF=now.strftime('%Y-%m-%d %H:%M'),
            scan=scan,
            output='raw'
        )
        if scan == 'FixedPoint':
            # instance.datos = instance.datos.resample('30s').mean()
            instance.raw = instance.raw.resample('30s').mean()
            instance.datosInfo = instance.datosInfo.resample('30s').mean()

        plotting(instance, scan, textSave='_48h')

        #24 Horas

        plotting(instance, scan, hora=24, textSave='_24h')

        #12 Horas
        plotting(instance, scan, hora=12, textSave='_12h')

        # #6 Horas
        # plotting(instance, scan, hora=6,textSave='_6h')

        #3 Horas
        plotting(instance, scan, hora=3, textSave='_3h')

    # except:
    #     import traceback
    #     traceback.print_exc()
    #     pass

if _name_ == '_main_':
    p = .Process(target=lidar_cron, args=(kind,), name="r")
    p.start()
    for i in xrange(300):
        time.sleep(1) # wait 60 seconds to kill process
        if not p.is_alive():
            p.terminate()
    p.terminate()
    p.join()


# def f(x):
#     return x*x
#
# if _name_ == '_main_':
#     p = Pool(5) # usar 5 núcleos
#     print(p.map(f, [1, 2, 3]))
#     p.join()
