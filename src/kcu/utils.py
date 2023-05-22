"""
Utils
"""
import os
import math
import typing
import psycopg2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from sqlalchemy import create_engine


def get_default_backend_config() -> typing.Dict:
    """
    Returns a dictionary of default connection settings to the SQL backend

    Returns:
        typing.Dict: dicitonary containing connection values
    """
    return {
        "host": os.getenv("SQL_HOST"),
        "port": os.getenv("SQL_PORT"),
        "db": os.getenv("SQL_DB"),
        "dbtype": "postgresql",
        "user": os.getenv("SQL_USER"),
        "password": os.getenv("SQL_PASSWORD"),
    }


def get_sql_url(backend: typing.Dict) -> str:
    if backend["dbtype"] == "postgresql":
        return "postgresql+psycopg2://" + backend["host"] + ":" + str(backend["port"]) + "/" + backend["db"]


def get_sql_engine(backend: typing.Dict):
    return create_engine(get_sql_url(backend))


def get_pyspark_driver(driver_name: str) -> str:
    """
    Returns the driver-string of the given backend

    Args:
        driver_name (str): the name of the backend
    Returns:
        str: driver string
    """
    if driver_name == "sqlite":
        return "org.sqlite.JDBC"
    elif driver_name == "postgresql":
        return "org.postgresql.Driver"


def get_pyspark_session(backend_type: str=None) -> SparkSession:
    """
    Returns Sparksession for given backend

    Args:
        backend_type (str): backend

    Returns:
        SparkSession: the built SparkSession
    """
    sess = SparkSession \
        .builder \
        .appName("appname")

    pluginpath = os.path.abspath(os.path.dirname(__file__)) + "/plugins"
    if backend_type == "sqlite":
        sess = sess.config(
                        "spark.jars",
                        "{}/sqlite-jdbc-3.34.0.jar".format(pluginpath)) \
                    .config(
                        "spark.driver.extraClassPath",
                        "{}/sqlite-jdbc-3.34.0.jar".format(pluginpath))
    elif backend_type == "postgresql":
        sess = sess.config("spark.jars", pluginpath + "/postgresql-42.5.1.jar")

    return sess.getOrCreate()


def get_df_from_backend(table: str, backend: typing.Dict, sess: SparkSession) -> DataFrame:
    """
    Returns dataframe from backend

    Args:
        table (str): Table name
        backend: (typing.Dict): backend connection settings
        sess (SparkSession): Spark session
    Returns:
        DataFrame: returned dataframe
    """
    df = sess.read.format('jdbc').options("driver", get_pyspark_driver(backend["dbtype"]))

    if backend["dbtype"] == "sqlite":
        df = df.options(dbtable=table,
                        url='jdbc:' + "sqlite:///" + backend["filepath"])
    elif backend["dbtype"] == "postgresql":
        df = df.option("url", "jdbc:postgresql://" + backend["host"] + ":" + str(backend["port"]) + "/" + backend["db"]) \
        .option("dbtable", table) \
        .option("user", backend["user"]) \
        .option("password", backend["password"])

    return df.load()


def report_dataframe(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    # profile = ProfileReport(
    #    dataset, title="Dataset", html={"style": {"full_width": True}}, sort=None
    # )
    return None


def correlation_matrix(x_data):
    """_summary_

    Args:
        X (_type_): _description_
    """
    plt.rcParams["figure.figsize"] = (12, 6)
    _, axis = plt.subplots()
    corr_ = x_data.corr()
    mask = np.triu(np.ones_like(corr_.to_numpy(), dtype=bool))
    sns.heatmap(
        corr_,
        annot=True,
        linewidths=0.5,
        fmt=".2f",
        ax=axis,
        mask=mask,
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )

    axis.set_title("Feature correlations")
    axis.set_xticklabels(
        axis.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    axis.set_yticklabels(
        axis.get_yticklabels(), rotation=45, horizontalalignment="right"
    )
    plt.show()


def correlation_for_y(data_x, data_y, feats_per_line=25):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        feats_per_line (int, optional): _description_. Defaults to 25.
    """
    plt.rcParams["figure.figsize"] = (12, 6)
    feats = len(data_x.columns)
    for i in data_y.unique():
        fig, axis = plt.subplots()
        corr_ = (
            pd.concat([data_x, data_y], axis=1, ignore_index=True)
            .corr()[-1][0:-1]
            .to_numpy()
            .reshape(-1, feats_per_line)
        )
        im1 = axis.imshow(
            corr_, cmap="PiYG", interpolation="nearest", vmin=-1.0, vmax=1.0
        )
        axis.set_title("Class " + str(i) + " feature correlations.")
        for j in range(feats_per_line):
            for k in range(int(feats / feats_per_line) + 1):
                if (j + (k * 25)) < feats:
                    axis.text(
                        j,
                        k,
                        j + (k * feats_per_line),
                        ha="center",
                        va="center",
                        color="gray",
                    )
        fig.colorbar(im1, orientation="horizontal")
        axis.set_axis_off()
        # plt.savefig("correlations_" + str(i) + ".svg")
        plt.show()


def multiple_correlation(data, data_z):
    """_summary_

    Args:
        data (_type_): _description_
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    # https://stackoverflow.com/questions/55369159/how-to-perform-three-variable-correlation-with-python-pandas
    cor = data.corr()
    new_corr = np.zeros_like(cor.to_numpy())
    # Independent variables
    for num1, data_x in enumerate(data.columns):
        for num2, data_y in enumerate(data.columns[:num1]):
            if data_x != data_y:
                xz_ = cor.loc[data_x, data_z]
                yz_ = cor.loc[data_y, data_z]
                xy_ = cor.loc[data_x, data_y]
                rxyz = math.sqrt(
                    (abs(xz_**2) + abs(yz_**2) - 2 * xz_ * yz_ * xy_)
                    / (1 - abs(xy_**2))
                )
                r_2 = rxyz**2
                new_corr[num1, num2] = r_2
    return new_corr
