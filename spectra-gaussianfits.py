
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from reproject import reproject_interp
import regions
from spectral_cube import SpectralCube
from lmfit.models import GaussianModel
from numpy import exp, pi, sqrt
from matplotlib import rc

# Setup for better appearance in plots
rc('font', **{'family': 'serif', 'serif': ['serif']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})

# Path to the data cube
cii_cube_path = "../../Observations_AllFiles/upGREAT/PCA/NGC7538_CII_merged_PCA.fits"
regions_path = '../ForAverageSpectra.reg'

# Function to read and process spectral cube
def read_and_process_cube(cube_path, regions_path):
    cube = SpectralCube.read(cube_path)
    region_list = regions.read_ds9(regions_path)
    print(region_list)

    intensity = []
    velocity = []

    for region in region_list:
        sub_cube = cube.subcube_from_regions([region])
        spectrum = sub_cube.mean(axis=(1, 2))
        intensity.append(spectrum.to_value())
        velocity.append(spectrum.spectral_axis.to_value("u.km/u.s"))
    
    return intensity, velocity

# Gaussian function
def mygauss(x, amp, cen, wid):
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

# Function to fit Gaussian models
def fit_gaussian_models(intensity, velocity):
    model = (GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') +
             GaussianModel(prefix='g3_') + GaussianModel(prefix='g4_'))
    params = model.make_params(g1_amplitude=5, g1_center=-65, g1_sigma=0.1,
                               g2_amplitude=5, g2_center=-58, g2_sigma=0.1,
                               g3_amplitude=5, g3_center=-10.4, g3_sigma=0.1,
                               g4_amplitude=5, g4_center=-48, g4_sigma=0.1)
    result = model.fit(intensity[0], params, x=velocity[0])
    return result

# Function to plot results
def plot_results(velocity, intensity, result):
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0)

    # Plot spectrum and fit
    ax[0].step(velocity[0], intensity[0], linewidth=1, color='black', label=r'[CII] 158 $\mu$m', alpha=0.85)
    for i, color, label in zip(range(1, 5), ['blue', 'green', 'red', 'magenta'], ['1st', '2nd', '3rd', '4th']):
        ax[0].fill_between(velocity[0], mygauss(velocity[0], result.params[f"g{i}_amplitude"].value,
                                                result.params[f"g{i}_center"].value, result.params[f"g{i}_sigma"].value),
                           facecolor=color, edgecolor=None, linestyle='dashed', linewidth=1, label=label, alpha=0.5)

    ax[0].set_xlim(-90, 0)
    ax[0].set_ylabel(r"$T_\mathrm{mb}$ [K]")
    ax[0].get_yaxis().set_label_coords(-0.2, 0.5)

    # Residual plot
    residual = intensity[0] - result.best_fit
    ax[1].scatter(velocity[0], residual, color='black', s=8)
    plt.xlim(-90, 0)
    plt.xlabel(r"$V_\mathrm{LSR}$ [km s$^{-1}$]")
    plt.ylabel(r"Residual [K]")
    ax[1].get_yaxis().set_label_coords(-0.2, 0.5)

    plt.tight_layout(h_pad=0.0, w_pad=0.1)
    plt.savefig("NGC7538_AverageCII_GaussianFitted.pdf", dpi=600)
    plt.show()


read_and_process_cube(cii_cube_path, regions_path)
fit_gaussian_models()
plot_results()
