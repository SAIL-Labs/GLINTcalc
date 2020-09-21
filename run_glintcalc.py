"""
Calculate signal to noise ratio, etc. for upgraded GLINT instrument.
Uses glintcalc.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from glintcalc import *
plt.ion()


##### Set a bunch of values #####
magnitude = 5

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
mag0flux = 1e10 #ph/um/s/m^2
star_flux = mag0flux * 2.5**-magnitude

wavelength = 1.6 # microns
bandwidth = 0.05 # microns #TODO - Treat chromaticity properly (photonic)

# Collecting area:
pupil_fraction = 0.5
pupil_area = np.pi*4**2 * pupil_fraction

# Wavefront properties
deltaphi_sig = 0.02 # RMS wavefront across baseline in microns. #TODO - Make a function of wavefront
deltaI_sig = 0.05 # RMS difference in injection between two inputs of baseline. #TODO - Make a function of wavefront

# Throughputs:
scexao_throughput = 0.2
injection_efficiency = 0.2 #TODO - Make a function of wavefront

# Companion contrast
contrast = 1e-6

# Detector properties
read_noise = 0.5 # e-
QE = 0.8
int_time = 3600 # seconds

print('Using deltaphi_sig = %f and deltaI_sig = %f' % (deltaphi_sig, deltaI_sig))


##### Do some calculations #####

g = glintcalc(wavelength = wavelength)
av_null = g.get_null_vals_MC(deltaphi_sig, deltaI_sig, num_samps=100000, show_plot=True)
print('Average null depth: %f' % av_null)

g.plot_null_dphi(deltaI_sig=deltaI_sig, max_dphi=0.1)

chrom_null = g.get_chromatic_null(deltaphi_sig, deltaI_sig, bandwidth=bandwidth, show_plot=True)
print('Chromatic average null depth: %f' % chrom_null)

null_depth = chrom_null
nulled_comp_snr = g.get_snr(photon_flux=star_flux, bandwidth=bandwidth, contrast=contrast, null_depth=null_depth,
                            throughput=scexao_throughput*injection_efficiency, pupil_area=pupil_area,
                            int_time=int_time, read_noise=read_noise, QE=QE, num_pix=1)
