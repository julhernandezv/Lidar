# -*- coding: utf-8 -*-

import pandas as pd
import MySQLdb

class sqldb:
    '''Class for manipulating databases'''
    def __init__(self,host="localhost",user="jhernandezv",passwd="aF05wnXC;",dbname="secop",table="PyWeb"):
        self.host   = host
        self.user   = user
        self.passwd = passwd
        self.dbname = dbname
        self.table  = table

    def mysql_desc_table(self,table):
        return list(pd.DataFrame(np.matrix(self.mysql_query("describe %s;"%table))).set_index(0).index)

    def query_db (self, where=None, columns='*',  **kwargs):

        """Querys on databases

        columns = list or array. Default is all in table
        where   = tuple (code, value)  or list of tuples for multiple conditionals  """

        self.conn_db 		= MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
        #else:		self.conn_db 		= psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" %(self.dbname,self.user,self.host,self.paswwd))

        query = "SELECT {} FROM {}".format( str(list(columns)).strip("[]").replace("u'","").replace("'",""), self.table )

        if where is not None:
            query += "WHERE {} in ('{}')".format( where[0],where[1] ) if isinstance(where, tuple) else ' '.join( [ "%s in ('%s') AND " %(w[0],w[1]) for w in where ] ) [:-5]

        df = pd.read_sql( query, self.conn_db )
        self.conn_db.close ()

        return df

    def read_sql(self,sql):
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        df = pd.read_sql(sql,conn_db)
        conn_db.close()
        return df

    def insert_into(self,df):

        for ix in df.index.values:
            print (ix)

            serie 		= df.loc[ix].dropna()

            datos 		= ', '.join(map(lambda x: "'{}'".format(serie[x]) , serie.index.values ) )
            columns	= ', '.join(map(lambda x: '%s' %x, serie.index.values ))

            query		= "INSERT INTO %s (%s) VALUES (%s)" %(self.table, columns, datos)

            conn_db 	= MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
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
