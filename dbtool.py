# -*- coding:utf8 -*-
import sqlite3

'''
sqlite3数据操作简易封装
'''


class DBHP():

    def __init__(self, db_name=None):
        self.conn = sqlite3.connect(db_name if db_name else 'linkdata.cf')
        self.cursor = self.conn.cursor()
        print("初始化打开数据库成功")

    '''
    创建表格
    @:param table_name 表名
    @:param field_list 字段列表,例如：["name","age","gender"]
    @:return 
    '''

    def create_tables(self, table_name: str, field_list: list) -> bool:
        try:
            fields = ",".join([field + " TEXT" for field in field_list])
            sql = f"CREATE TABLE {table_name} ({fields});"
            self.cursor.execute(sql)
            self.conn.commit()
            return True
        except Exception as ex:
            print("创建表出错，错误信息：", str(ex))
            return False
    def create_link_table(self):
        sql = '''
                    CREATE TABLE IF NOT EXISTS `link`(
                       `id` INTEGER PRIMARY KEY,
                       `url` VARCHAR(100) NOT NULL,
                       `num` INTEGER  NOT NULL,
                       `snum` VARCHAR(100) NOT NULL,
                       `datetime` TIMESTAMP default (datetime('now','localtime'))
                    );
                    '''

        self.cursor.execute(sql)

    '''
    插入数据，根据传入的数据类型进行判断，自动选者插入方式
    @:param table_name 表名
    @:param data 要插入的数据
    '''

    def insert_data(self, table_name: str, data) -> bool:
        try:
            if isinstance(data, list):
                for item in data:
                    keys = ",".join(list(item.keys()))
                    values = ",".join([f"'{x}'" for x in list(item.values())])
                    sql = f"INSERT INTO {table_name} ({keys}) VALUES ({values});"
                    self.cursor.execute(sql)
            elif isinstance(data, dict):
                keys = ",".join(list(data.keys()))
                values = ",".join([f"'{x}'" for x in list(data.values())])
                sql = f"INSERT INTO {table_name} ({keys}) VALUES ({values});"
                self.cursor.execute(sql)
            return True
        except Exception as ex:
            return False
        finally:
            self.conn.commit()

    '''
    查询数据
    @:param 要查询的sql语句
    '''

    def query_data(self, sql: str) -> list:
        try:
            self.conn.execute(sql)
            results = self.cursor.fetchall()
            return results
        except Exception as ex:
            return []

    '''
    关闭数据库连接
    '''

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
        except Exception as ex:
            raise Exception("关闭数据库连接失败")

# data=[
#     {"name":"张三","age":"23"},
#     {"name":"张三","age":"23"},
#     {"name":"张三","age":"23"}
# ]

db=DBHP(db_name="linkdata.cf")
# db.create_link_table()
# db.create_tables("link",['id', 'url', 'datetime', 'num'])
# # db.insert_data("stu",data)
for item in db.query_data("select * from link"):
    print(item)
db.close()