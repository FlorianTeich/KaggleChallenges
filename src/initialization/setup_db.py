import os

import pandas as pd
import sqlalchemy


def setup_db_titanic() -> int:
    try:
        print("Copying Titanic Dataset into SQL-DB")
        titanic = pd.read_csv(
            "https://raw.githubusercontent.com/rolandmueller/titanic/main/titanic3.csv"
        )
        engine = sqlalchemy.create_engine(
            "postgresql+psycopg2://"
            + os.getenv("SQL_USER")
            + ":"
            + os.getenv("SQL_PASSWORD")
            + "@"
            + os.getenv("SQL_HOST")
            + ":"
            + os.getenv("SQL_PORT")
            + "/"
            + os.getenv("SQL_DB")
        )
        titanic.to_sql("titanic", engine, if_exists="replace")
        print("done.")
    except Exception as exc:
        print(exc)
        return -1
    return 1
