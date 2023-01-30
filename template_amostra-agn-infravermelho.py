# -*- coding: utf-8 -*-
"""
PROGRAM TO PLOT ALL THE SPECTRA WITH REDSHIFT CORRECTION AND THE RESPECTIVE
TEMPLATE SPECTURM OF THE SAMPLE AVAILABLE IN THE SPITZER/IRS ATLAS PROJECT
 
Returns one image with all the sources

The sources were extracted from A. Hernán-Caballero and E. Hatziminaoglou
(2011). Site: http://www.denebola.org/atlas/?p=data

INPUT: directory of the sources and their final designation
       directory to save the plots
       path of the ATLAS template
       name of the final image
       title of the plot
       range of the x and y axis

OUTPUT: plots in .png and .pdf

Code written by Carla Martinez Canelo - Jun/2021.
Last modification -- 
"""
################################# BEGIN  #####################################
import numpy as np
import glob
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#### INPUT

# Change the files directory if necessary 
# destiny --> local where results will be saved
# Change the files name if necessary 

# directory of the sources
source = '/home/carla/Desktop/vitor/espectros_originais/MIR_AGN1_z/'
sources = source + '*_z.dat' # final designation of the sources

# directory to save the plots
destiny = ''

# path of the ATLAS template
med_spectrum = '/home/carla/Desktop/vitor/ATLAS_templates/MIR_AGN1.dat'

# name of the final image
img_name = 'all_spectra_AGN1'

# title of the plot
title_plot = 'MIR_AGN1 subsample'

# range of the x and y axis  [xmin, xmax, ymin, ymax]
limits_plot = [5, 15, 0.3, 4.1]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Creating a list with the objects from the chosen path
S = np.array(glob.glob(sources), dtype='U')
S.sort()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#### Loading the sample's template spectrum

# w = wavelength, f = flux, ferr = flux error
w, f, fe = np.loadtxt(med_spectrum, unpack=True, usecols=[0,1,2])

# Selecting the flux at 7um to perform the normalization
 
w_2 = w[w>=7]  # w_2[0] is the first wavelength >= 7um
    
i2 = np.where(w == w_2[0]) # i2 is the index of w_2[0] in the total spectrum
    
f7 = f[i2] # f7 is the flux at 7um


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#### Loading the observational data and plotting

# Plot parameters
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('legend', fontsize=8)  # fontsize of the legend

fig = plt.figure(figsize=(7,5))

plt.title(title_plot)
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'Normalized flux intensity at 7 $\mu$m')
plt.axis(limits_plot)

# ---------------------------------------------------------------------------
# Plotting each spectrum of the sample normalized at 7 um
# (same procedure performed with the template spectrum)

for i, j in enumerate(S):

    wavel, flux, ferr = np.loadtxt(j, unpack=True)
    
    wave_2 = wavel[wavel>=7]
    
    idx = np.where(wavel == wave_2[0])
    
    flux7 = flux[idx]

    # Plot the normalized spectrum        
    plt.plot(wavel, flux/flux7, ls='-', color='#bababa', lw=1)

# ---------------------------------------------------------------------------
# Plotting the template spectrum
    
plt.plot(w, f/f7, ls='-', color='#f70046', lw=2)
 
# ---------------------------------------------------------------------------
# Show and save the plot
    
plt.tight_layout()
   
plt.show()    
    
fig.savefig(img_name+'.pdf', dpi=fig.dpi)
fig.savefig(img_name+'.png', dpi=fig.dpi)

#################################### END #####################################
