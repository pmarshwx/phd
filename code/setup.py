from __future__ import print_function, division
import distutils.sysconfig
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.install_data import install_data
import numpy as np
import os, sys

# Setup Variables
dirname = 'disser'
packages = ['disser', 'disser.verification', 'disser.stat_tools',
            'disser.misc', 'disser.calibration', 'disser.viz']
setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, dirname))
module_path = distutils.sysconfig.get_python_lib()
data_files = {dirname: ['data/*']}

import __verfuncs__ as verfuncs
verfuncs.write_git_version()
ver = verfuncs.get_version()
sys.path.pop()

setup(
    name                    = 'disser',
    version                 = ver,
    description             = 'Dissertation Code/Utilities',
    author                  = 'Patrick Marsh',
    author_email            = 'patrick.marsh@noaa.gov',
    url                     = '',
    download_url            = '',
    packages                = packages,
    package_data            = data_files,
    scripts                 = [],
    cmdclass                = {},
    ext_modules             = [],   )

