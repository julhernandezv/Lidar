# -*- coding: utf-8 -*-
"""
    This code have to be running into torresiata@miel for ordering
    fiexed point data warehouse.
"""


import os, glob, locale
import datetime as dt


locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))


baseDir = '/mnt/ALMACENAMIENTO/LIDAR/'
fpPath = baseDir +'LastFiles/Fixed_Point/'
# smPath = baseDir +'LastFiles/Scanning_Measurements/'

fpFiles = glob.glob('%sRM*' %fpPath)
# smFiles = glob.glob('%s*' %smPath)

#Moving Fixed Point Files
for fp in fpFiles:
    f = fp.strip(fpPath)
    f = '{}-{}-{}'.format(
                    f[:2],
                    f[2:3].replace('A','10'
                        ).replace('B','11'
                        ).replace('C','12'),
                    f[3:5])
    day = dt.datetime.strptime(f, '%y-%m-%d')
    path = '{}Fixed_Point/{}/'.format(
                    baseDir,
                    day.strftime('%Y%m%d'))
    print path
    if not os.path.exists(path):
        os.makedirs(path)
    os.system('mv {} {}'.format(fp,path))
    os.system('chmod -R ugo+rx %s' %path)
# #Moving Scanning Folders
# for sm in smFiles:
#     s   = sm.strip(smPath)
#     day = dt.datetime.strptime(s[4:10], '%y%m%d')
#     path = '{}Scanning_Measurements/{}/'.format(
#                     baseDir,
#                     day.strftime('%Y%m%d'))
#     print path
#     if not os.path.exists(path):
#         os.makedirs(path)
#     os.system('mv {} {}'.format(sm,path))
