import os
import numpy as np
import __verfuncs__
import verification
import stat_tools
import misc
import calibration
import viz

__all__ = ['verification', 'stat_tools', 'misc', 'calibration', 'viz']

__version__ = __verfuncs__.get_version()

# Setup the Mask
mask = np.load(
        os.path.join(os.path.dirname(__file__), 'data/mask.npz'))['mask']
