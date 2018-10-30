# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
import sys, os, time, locale
import multiprocessing as mp
from functools import wraps
from lidar.utils.utils import LoggingPool


baseDir = os.path.dirname(os.path.abspath(__file__))
print sys.argv
kind = sys.argv[1]

baseDir = os.path.join( baseDir, kind )
print baseDir
if not os.path.exists( baseDir ):
    os.makedirs( baseDir )

os.chdir ( baseDir )

from lidar.lidar import Lidar
locale.setlocale(locale.LC_TIME, ('en_GB','utf-8'))

now = dt.datetime.now() #- dt.timedelta(days=8)



def plots (obj,  scan, **kwargs):

    kwgs = dict(
        height=4.5 if scan  != 'Zenith' else 6,
        operational=True,
        # cla=True
    )
    kwgs.update(kwargs)


    obj.plot( output='LVD', **kwgs )

    obj.plot( output='RCS', **kwgs )

    obj.plot( output='fdfLn(RCS)', totalSignal=True, **kwgs)



def plotting(element, scan,  **kwargs):
    if 'hora' in kwargs.keys():
        element.raw = element.raw[
                        (now-dt.timedelta(hours=kwargs.pop('hora'))
                            ).strftime('%Y-%m-%d %H:%M'):]
    plots(element, scan, **kwargs)
    if scan != 'FixedPoint':
        plots(element, scan,
            dates=element.raw.index,
            scp=False,
            makeGif=True,
            **kwargs)

def worker_wrapper(arg):
    args, kwargs = arg
    return plotting(*args, **kwargs)

def main(scan):
    print 'cpu_count() = %d\n' % mp.cpu_count()

    PROCESSES = 5
    print 'Creating pool with %d processes\n' % PROCESSES
    pool = LoggingPool(processes=PROCESSES)
    print 'pool = %s' % pool


    instance =   Lidar(
        fechaI=(now-dt.timedelta(hours=48)).strftime('%Y-%m-%d %H:%M'),
        fechaF=now.strftime('%Y-%m-%d %H:%M'),
        scan=scan,
        output='raw',
        path='CalidadAire/Lidar/'

    )
    if scan == 'FixedPoint':
        # instance.datos = instance.datos.resample('30s').mean()
        instance.raw = instance.raw.resample('30s').mean()
        instance.datosInfo = instance.datosInfo.resample('30s').mean()

    args = (instance, scan)
    TASKS = [(args, dict(hora=h,textSave='_%dh' %h)) for h in [48,24,12,6,3]]

    pool.map(worker_wrapper, TASKS)

    print 'Testing close():'
    for worker in pool._pool:
        assert worker.is_alive()

    pool.close()
    pool.join()

    for worker in pool._pool:
        assert not worker.is_alive()

    print '\tclose() succeeded\n'

    print  "Time: {}".fromat( (dt.datetime.now()-start).total_seconds() )




if __name__ == '__main__':
    main(kind)
    # p = Process(target=lidar_cron, args=(kind,), name="r")
    # print "time start {}".format(dt.datetime.now())
    # p.start()
    # time.sleep(300)
    # print "time running {}".format(dt.datetime.now())
    # while True:
    #     if not p.is_alive():
    #         p.terminate()
    #         break
    # print "time end {}".format(dt.datetime.now())
    # p.join()
