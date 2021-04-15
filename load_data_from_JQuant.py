import psycopg2
from jqdatasdk import *
from utils import *
import pandas as pd
import time
import datetime
auth('18795642715', 'Litianyi123')


def get_all_data(stocks, start_date, end_date, frequency):
    df_all = get_price(stocks, start_date=start_date, end_date=end_date, frequency=frequency, fields=None,
                       skip_paused=False, fq='none')
    return df_all['close']


def create_table(curs, conn, table_name, columns):
    sql = 'CREATE TABLE ' + table_name + '('
    sql += 'Time timestamp'
    for column in columns:
        sql += ', ' + column[-4:] + column[:6] + ' FLOAT8'
    sql += ')'
    curs.execute(sql)
    conn.commit()


def commit_row_data(curs, conn, df, table_name, now):
    df_total = df.reset_index()
    df_total.rename(columns={'index': 'Time'}, inplace=True)
    keys = '('
    values = '('
    for i in range(len(df_total.columns)):
        if i == 0:
            keys += df_total.columns[i] + ', '
            values += "'" + now.strftime("%Y-%m-%d %H:%M:%S") + "', "
        elif i == len(df_total.columns) - 1:
            keys += df_total.columns[i][-4:] + df_total.columns[i][:6] + ')'
            if str(df_total[df_total.columns[i]].values[0]) == 'nan':
                values += str(0) + ')'
            else:
                values += str(df_total[df_total.columns[i]].values[0]) + ')'
        else:
            keys += df_total.columns[i][-4:] + df_total.columns[i][:6] + ', '
            if str(df_total[df_total.columns[i]].values[0]) == 'nan':
                values += str(0) + ', '
            else:
                values += str(df_total[df_total.columns[i]].values[0]) + ', '

    insert_sql = 'insert into ' + table_name + keys + 'values ' + values
    curs.execute(insert_sql)
    # 提交数据
    conn.commit()


def load_once(now, last_price=None, price_name='price', return_name='return'):
    conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195',
                            port='5432')
    curs = conn.cursor()
    total_index = list(np.load('columns.npy', allow_pickle=True))
    print(len(total_index))
    time.sleep(10000)
    dt = datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
    now = now - dt
    df_total = get_all_data(total_index, frequency='minute', start_date=now, end_date=now)
    print(df_total)
    time.sleep(1000)
    if last_price is None:
        last_price = df_total.values.copy()
    temp = df_total.values.copy()
    if len(temp) == 0:
        return None
    else:
        commit_row_data(curs, conn, df_total, price_name, now)

        temp1 = df_total.values / last_price
        df_total.iloc[:, :] = temp1
        last_price = temp
        commit_row_data(curs, conn, df_total, return_name, now)
        return last_price

'''
total_index = list(np.load('columns.npy'))
conn = psycopg2.connect(database='postgres', user='kline', password='kline@tushare', host='47.107.103.195', port='5432')
curs = conn.cursor()
# create_table(curs, conn, 'Price', total_index)
# create_table(curs, conn, 'Return', total_index)

last_price = None
while True:
    try:
        start_time = time.time()
        hours = datetime.timedelta(days=0, hours=7)
        now = datetime.datetime.now() - hours
        last_price = load_once(now=now, last_price=last_price)
        end_time = time.time()
        print(end_time - start_time)
        time.sleep(60 - (end_time - start_time))
    except KeyboardInterrupt:
        curs.close()
        break
'''


