import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd


def save_fig(image_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save the figure to the folder"""
    path = os.path.join(image_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def fetch_housing_data(housing_url, housing_path):
    """Fetch dataset for chapter 2"""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """Load the dataset"""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
