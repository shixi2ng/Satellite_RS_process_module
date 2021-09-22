import gdal
import sys
import collections
import snappy
from snappy import PixelPos, Product, File, ProductData, ProductIO, ProductUtils, ProgressMonitor
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
import datetime
from datetime import datetime, date
import rasterio
import math
b = np.zeros((4, 4))
x = np.zeros((4, 4))
x[3, 2] = 1
print(x.sum(axis=0).sum(axis=0))
b[1, :] = x.sum(axis=0)
print(b)