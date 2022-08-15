import io
import os
from time import time
import numpy as np
import dill as pickle

import sqlite3
from sqlite3 import Error


def adapter(python_type):
    # adapt the Python type into an SQLite type
    out = io.BytesIO()
    np.save(out, python_type)
    return sqlite3.Binary(out.getvalue())


def converter(sqlite_object):
    # convert SQLite objects into a Python object
    return np.load(io.BytesIO(sqlite_object))


# https://www.sqlitetutorial.net/sqlite-python/creating-database/
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def initialize(db_filepath, table_name, adapter, converter):
    print("REGISTER ADAPTER/CONVERTER")
    sqlite3.register_adapter(np.ndarray, adapter)
    sqlite3.register_converter("array", converter)
    con = sqlite3.connect(db_filepath, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    
    # cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (idx integer, arr array)")
    
    # cur.execute("PRAGMA journal_mode = OFF")
    # cur.execute("PRAGMA synchronous = 0")
    cur.execute("PRAGMA cache_size = -51200") # kb
    # cur.execute("PRAGMA locking_mode = EXCLUSIVE")
    return con, cur


if __name__ == '__main__':
    data_dict = {
        "train": 1584663,
        "validation": 176074
    }

    file_type = "train"
    number = data_dict[file_type]

    file_name = "encoder_outputs.db"
    db_filepath = f"/fileserver-gamma/chaoting/ML/dataset/moses/raw/{file_type}/{file_name}"
    np_filefolder = f"/fileserver-gamma/chaoting/ML/dataset/moses/raw/{file_type}/encoder_outputs"

    table_name = "nparray"
    index_name = "arr_index"

    print("CREATE CONNECTION")
    create_connection(db_filepath)

    interval = 1000
    total_time = 0

    con, cur = initialize(db_filepath, table_name, adapter, converter)

    # con.execute(f"CREATE INDEX {index_name} ON {table_name} (idx)")
    
    # con.execute(f"SELECT arr FROM {table_name} WHERE idx = ?", (3,))
    
    t1 = time()
    cur.execute(f"SELECT idx, arr FROM {table_name} WHERE idx = ?", (10400,))
    cur.execute(f"SELECT idx, arr FROM {table_name} WHERE idx = ?", (190561,))
    t1 = time() - t1
    print(t1)

    t1 = time()
    cur.execute(f"SELECT idx, arr FROM {table_name} WHERE idx IN (?, ?, ?)", (10000, 1405142, 196570))
    t1 = time() - t1
    print(t1)

    records = cur.fetchall()
    for row in records:
        print(row)
    cur.close()
    print("???")
    exit()

    for i in range(number):
        tmp_time = time()
        arr = pickle.load(open(os.path.join(np_filefolder, f"{i+1}.pt"), "rb"))
        con.execute(f"INSERT INTO {table_name} (idx, arr) values (?, ?)", (i+1, arr))
        total_time += time() - tmp_time

        if (i+1) % interval == 0:
            print(f"FINISHED NUMBER: {i+1}  |  TOTAL TIME (s): {total_time}  |  AVERAGE TIME (s): {total_time/(i+1)}")

    con.commit()
    # cursor = con.execute(f"SELECT * FROM {table_name}")
    # print(cursor.fetchall())
    cur.close()

    print(f"TOTAL TIME (s): {total_time}")
