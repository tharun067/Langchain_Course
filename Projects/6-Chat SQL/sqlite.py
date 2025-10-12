import sqlite3


### Connect to sqlite database
connection = sqlite3.connect("student.db")

### Create a cursor to insert record, create table
cursor = connection.cursor()

### Create a table
table_info = """
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT)"""
cursor.execute(table_info)

### Insert record
cursor.execute("insert into STUDENT values ('John', '10th', 'A', 85)")
cursor.execute("insert into STUDENT values ('Alice', '10th', 'B', 90)")
cursor.execute("insert into STUDENT values ('Bob', '9th', 'A', 78)")
cursor.execute("insert into STUDENT values ('Eve', '9th', 'B', 88)")
cursor.execute("insert into STUDENT values ('Charlie', '10th', 'A', 92)")

### Display all records
print("The inserted records are:")
data = cursor.execute("select * from STUDENT")
for row in data:
    print(row)


### Commit the changes in database
connection.commit()
connection.close()