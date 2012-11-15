from __future__ import print_function, division
import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_data import install_data
import numpy as np
import os, sys

# Setup Variables
dirname = 'disser'
packages = ['disser']
setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, dirname))
module_path = distutils.sysconfig.get_python_lib()
include_files = ['../README.md']
data_files = [(os.path.join(module_path, dirname), include_files)]

import version
version.write_git_version()
ver = version.get_version()
sys.path.pop()

setup(
    name                    = 'disser',
    version                 = ver,
    description             = 'Dissertation Code/Utiilities',
    author                  = 'Patrick Marsh',
    author_email            = 'patrick.marsh@noaa.gov',
    url                     = '',
    download_url            = '',
    packages                = packages,
    data_files              = data_files,
    scripts                 = [],
    cmdclass                = {},
    ext_modules             = [],   )

