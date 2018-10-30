# -*- coding: utf-8 -*-

import traceback
import multiprocessing as mp
from multiprocessing.pool import Pool

################################################################################
# Decorator for logging errors
# Shortcut to multiprocessing's logger
def error(msg, *args):
    return mp.get_logger().error(msg, *args)

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
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)

    def map(self, func, iterable, chunksize=None ):
        return Pool.map(self, LogExceptions(func), iterable, chunksize)


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
