# -*- coding: utf-8 -*-

import numpy as np
import traceback
import multiprocessing as mp
import logging
import logging.handlers
# from functools import wraps
from multiprocessing.pool import Pool

from matplotlib.pyplot import register_cmap
from matplotlib import colors
################################################################################
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    cdict = {'red': [],'green': [],'blue': [],'alpha': []}

    regIndex = np.linspace(start, stop, 257)
#
    shiftIndex = np.hstack([

        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(regIndex, shiftIndex):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)

    register_cmap(cmap=newcmap)
    return newcmap

################################################################################
def listener_configurer(name,getLogger=None):
    if getLogger is None:
        getLogger = __name__
    root = logging.getLogger(getLogger)
    h = logging.handlers.RotatingFileHandler('{}.log'.format(name), 'a', 3000, 10)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    return root

# Decorator for logging errors
# Shortcut to multiprocessing's logger
def error(msg, *args):
    pname    = mp.current_process().name
    logger  = listener_configurer(name='Lidar',getLogger=pname)
    # logger  = logging.getLogger(name) #mp.get_logger() #log_to_stderr()
    # h = logging.handlers.RotatingFileHandler('Lidar.log', 'a', 3000, 10)
    # f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    # h.setFormatter(f)
    # logger.addHandler(h)
    logger.setLevel(logging.WARNING)
    return logger.error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

class LoggingPool(Pool):
    def apply_async(self, func, *args, **kwds):
        kwds['callback'] = kwds.get('callback',None)
        return Pool.apply_async(self, LogExceptions(func), *args, **kwds )

    def map(self, func, iterable, chunksize=None ):
        return Pool.map(self, LogExceptions(func), iterable, chunksize)

    def map_async(self, func, iterable, chunksize=None, callback=None ):
        return Pool.map_async(self, LogExceptions(func), iterable, chunksize, callback)
################################################################################
#Decorator for logfile
# def log(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start       = dt.datetime.now()
#
#         args_str    = ', '.join(args) + '\n'
#         kwargs_str  = ', '.join([ key for key  in kwargs.iterkeys()]) + '\n'
#         dx          = func(*args, **kwargs)
#         end         = dt.datetime.now()
#         func_str    = '%s, %d\n' %(func.__name__, (end-start).total_seconds() )
#         with open('lidarWrapper.log', 'a') as f:
#             f.write(func_str)
#             f.write(args_str)
#             f.write(kwargs_str)
#             f.close()
#         return dx
#     return wrapper
# #
