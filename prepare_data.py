import os 
import urllib.request
import zipfile
import tarfile

data_dir = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
