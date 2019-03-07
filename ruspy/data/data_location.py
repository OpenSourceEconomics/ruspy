"""
This module contains a function, which gives back the absolute path of the folder
data. This path can then be used to import and read the original data provided by
John Rust.
"""
import os


def get_data_storage():
    """
    :return: The absolute path of the folder data.
    """
    dirname = os.path.dirname(__file__)
    return dirname
