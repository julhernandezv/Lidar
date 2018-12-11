# -*- coding: utf-8 -*-

import pandas as pd
import MySQLdb
import psycopg2
from pymongo import MongoClient


class Database:
    '''Class for manipulating databases'''

    dbArgs = dict(
        kindDb='mysql',
        port=None,
        table=None,
        collection=None,
        noId=True,
    )

    def __init__(self, host, user, password, dbname, **kwargs ):

        self.host       = host
        self.user       = user
        self.password   = password
        self.dbname     = dbname

        for kw in self.dbArgs.keys():
            setattr(self, kw, kwargs.pop(kw, self.dbArgs[kw]))


    def mysql_desc_table(self,table=None):
        if table is None:
            table = self.table
        return list(
                pd.DataFrame(
                    np.matrix(
                        self.connect("describe %s;"%table)
                    )
                ).set_index(0).index
            )

    def query_sql (self, columns='*', where=None, **kwargs):

        """Querys on SQL databases

        columns = list or array. Default is all in table
        where   = tuple (code, value)  or list of tuples for multiple conditionals  """
        with dic as self.__dict__:
            for kw in dic.keys():
                setattr(self, kw, kwargs.pop(kw, dic[kw] ))

        query   = "SELECT {} FROM {}".format(
                    str(list(columns)).strip("[]").replace("u'","").replace("'",""),
                    self.table
                )

        if where is not None:
            if isinstance(where, tuple):
                query   += "WHERE {} in ('{}')".format( where[0],where[1] )

            else:
                query   += ' '.join(["%s in ('%s') AND " %(w[0],w[1]) for w in where ]
                        ) [:-5]

        return self.connect(query)


    def connect(self,query):

        if self.kindDb == 'mongodb':
            return self._connect_to_mongodb(query)

        elif self.kindDb == 'mysql':
            return self._connect_to_mysql(query)

        elif self.kindDb == 'postgresql':
            return self._connect_to_postgresql(query)
        else:
            raise ValueError('kindDb: {} invalid, choice anohter'.format(self.kindDb))


    def _connect_to_mongodb(self,query={}):

        if self.username is not None and self.password is not None:
            mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (
                            self.user, self.password, self.host,
                            self.port, self.dbname)
            conn = MongoClient(mongo_uri)
        else:
            conn = MongoClient(self.host, self.port)

        cursor = conn[self.dbname][self.collection].find(query)
        df =  pd.DataFrame(list(cursor))
        if self.noId:
            del df['_id']
        return df


    def _connect_to_mysql(self,query):
        conn_db     = MySQLdb.connect(
                        self.host,
                        self.user,
                        self.password,
                        self.dbname
                    )
        df = pd.read_sql(query, conn_db)
        conn_db.close()
        return df

    def _connect_to_postgresql(self,query):
        conn_db    = psycopg2.connect(
                        "dbname='{}' user='{}' host='{}' password='{}'".format(
                            self.dbname,
                            self.user,
                            self.host,
                            self.paswword
                        )
                    )
        df = pd.read_sql(query, conn_db)
        conn_db.close()
        return df


    def insert_into_sql(self,df):

        for ix in df.index.values:
            print (ix)

            serie 		= df.loc[ix].dropna()

            datos 		= ', '.join(map(lambda x: "'{}'".format(serie[x]) , serie.index.values ) )
            columns	= ', '.join(map(lambda x: '%s' %x, serie.index.values ))

            query		= "INSERT INTO %s (%s) VALUES (%s)" %(self.table, columns, datos)

            conn_db 	= MySQLdb.connect (self.host, self.user, self.password, self.dbname)
            cur 		= conn_db.cursor ()

            try:
            	cur.execute (query)
            	conn_db.commit()
            	print ('execute')
            	conn_db.close ()

            except	MySQLdb.IntegrityError as int_err:
                print (int_err)
                # if int_err.args[0] == 1062:
                #     print ('updating...')
                #     serie.drop('timestamp',inplace=True)
                #     datos_up 	= ', '.join(map(lambda x:'%s = ' %x + ('%.f' if x in column_type['int'] else '%.2f' if x in column_type['float'] else  "'%s'" ) %(serie[x].strftime('%Y-%m-%d') if x in column_type['date'] else  serie[x]) , serie.index.values ) )
                #
                #     query_up = "UPDATE %s SET %s  WHERE niuit = '%.f' " %(self.table,datos_up,serie.niuit)
                #     cur.execute (query_up)
                #     conn_db.commit()
                #     conn_db.close ()
                pass
