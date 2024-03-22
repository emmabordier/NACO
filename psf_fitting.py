import time
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from matplotlib import animation
from IPython.display import HTML
from IPython import display
import os
from glob import glob
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.datasets import make_100gaussians_image
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from photutils.segmentation import detect_threshold, detect_sources
from photutils.background import Background2D, MedianBackground
from matplotlib.patches import Rectangle
from astropy.io import ascii
from matplotlib.colors import LogNorm
import scipy.optimize as opt
#Needed to set up the IterativelySubtractedPSFPhotometry object
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup, BasicPSFPhotometry
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table, Column
from astropy.stats import mad_std
from photutils import DAOStarFinder
from photutils.centroids import centroid_com, centroid_quadratic
from photutils.centroids import centroid_1dg, centroid_2dg
from photutils import aperture_photometry
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAnnulus



def background_estimation(data):
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    return bkg

def magnitude_peak(m1,f1,f2):
    m2=m1+2.5*np.log10(f1/f2)
    delta_m=-2.5*np.log10(f2/f1)   #difference in magnitude between secondary and primary
    return m2,delta_m

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_vector(arrList, value):

    y,x = value
    a = arrList

    b = value

    idx_min = np.sum( (a-b)**2, axis=1, keepdims=True).argmin(axis=0)
    idx_min, a[idx_min]


    return idx_min, a[idx_min]

#FAKE SOURCES - Adding some "artifical" stars to the data to measure the detection limit
def stars(image, number, max_counts,flux_range, gain=1):

    from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image

    if flux_range==None:
        flux_range = [max_counts/8000, max_counts/50]     #defining flux min and max we want the fake stars to be
        # print('you\'re going in that loop')

    y_max, x_max = image.shape
    xmean_range = [0.45 * x_max, 0.55 * x_max]   #0.1
    ymean_range = [0.45 * y_max, 0.55 * y_max]
    xstddev_range = [2, 2]
    ystddev_range = [2, 2]
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])

    sources = make_random_gaussians_table(number, params,
                                          seed=12345)

    star_im = make_gaussian_sources_image(image.shape, sources)
    star_im = star_im+image

    return star_im, sources,max_counts, flux_range

# #centroid central star:
# centroid=centroid_1dg(data,mask=bpm_bool)

def find_peaks_image(master,dir_data):
    bkg=background_estimation(master)
    mean, median, std = sigma_clipped_stats(master, sigma=3.0)
    number_std=10.
    criteria=number_std*std
    threshold = bkg.background + criteria
    file_bpm=[file for file in glob(dir_data+"M*") if fits.open(file)[0].header["HIERARCH ESO PRO CATG"]=="MASTER_IMG_FLAT_BADPIX"][0]
    bpm=fits.open(file_bpm)[0]
    bpm_bool=bpm.data.astype(bool)
    tbl = find_peaks(master, threshold,mask=bpm_bool)


    #Checking that the region represents more than 1 pixel:

    for i in range (len(tbl['x_peak'])):
        if master[tbl['y_peak'][i]-2][tbl['x_peak'][i]-2]<np.median(threshold+criteria):
            #bad_peaks.append((tbl['y_peak'][i],tbl['x_peak'][i]))

            aper= CircularAperture((tbl['y_peak'][i],tbl['x_peak'][i]), 3.0)
            mask=aper.to_mask(method='exact').to_image(np.shape(master)).astype(bool)
            bpm_bool+=mask
    # print("There are",len(bad_peaks),"BAD PEAKS: ")

    #Adding the 1pixel regions to the mask
    # for i in bad_peaks:
    #     mask[i]=True

    tbl = find_peaks(master, threshold,mask=bpm_bool, box_size=25, border_width=30)
    tbl['peak_value'].info.format = '%.8g'

    positions_find = np.transpose((tbl['x_peak'], tbl['y_peak']))
    apertures_find = CircularAperture(positions_find, r=4.)

    plt.figure(figsize=(10,6))
    norm = ImageNormalize(stretch=SqrtStretch())

    # plt.subplot(1,2,1)
    plt.imshow(master, origin='lower',vmin=10,vmax=500)#norm=norm)#vmin = np.median(data[data<0]),vmax=0.1*np.max(data))
    apertures_find.plot(color='white', lw=1)
    for i in range (len(tbl['x_peak'])):
        plt.text(tbl['x_peak'][i]+10, tbl['y_peak'][i]+5,str(i+1),color='white',weight='bold',fontsize=6)
    # plt.title(source_name+": "+str(len(tbl['x_peak']))+" local peaks above threshold \n (background+"+str(number_std)+") \n Find Peaks Method")
    plt.show()

    return tbl

def star_detection(image,dir_data, roundlo, roundhi,sharphi=None,bpm_mask=False,xycoords=None):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    bkg=background_estimation(image)
    bkg_sigma = mad_std(image)
    number_std=5.

    file_bpm=[file for file in glob(dir_data+"M*") if fits.open(file)[0].header["HIERARCH ESO PRO CATG"]=="MASTER_IMG_FLAT_BADPIX"][0]
    bpm=fits.open(file_bpm)[0]
    bpm_bool=bpm.data.astype(bool)
    if bpm_mask==True:
        # bpm_bool[480:550, 480:550] = True
        # bpm_bool[507:524, 504:520] = False
        aper= CircularAnnulus((516,511), 5.0, 40.0)
        # aper= CircularAnnulus((330,220), 20.0, 1000.0)
        mask=aper.to_mask(method='exact').to_image(np.shape(image)).astype(bool)
        bpm_bool=bpm_bool+mask
#     print("threshold: ",np.median(bkg.background)+number_std*bkg_sigma)
#     print(bkg_sigma, std)
    dao = DAOStarFinder(fwhm=2., threshold=number_std*bkg_sigma,sharphi=sharphi, roundlo= roundlo, roundhi = roundhi, exclude_border=True)#,xycoords=xycoords)#,brightest=30) #sharphi=0.7  #roundness set so that only round objetcs are detected
    sources = dao(image - median ,mask=bpm_bool)
    # dao = DAOStarFinder(fwhm=2., threshold=np.max(bkg.background)+number_std*bkg_sigma,roundlo=-0.5, roundhi=0.5,exclude_border=True)   #roundness set so that only round objetcs are detected
    # sources = dao(image - median,mask=bpm_bool)
    for col in sources.colnames:
           sources[col].info.format = '%.5g'
    # sources['xcentroid'].info.format = '%.5g'
    # sources['ycentroid'].info.format = '%.5g'

    return sources,bpm

#ascii.write(sources, dir_reduced+'results_psf_fitting_'+source_name+'.dat', overwrite=True)

def print_detections(source_name,image,detection_list,bpm,vmin=None,vmax=None,extra_sources=None):
    positions_dao = np.transpose((detection_list['xcentroid'], detection_list['ycentroid']))
    apertures_dao = CircularAperture(positions_dao, r=3.)
    apertures_bpm = CircularAperture(np.argwhere(bpm.data!=0).tolist(), r=4.)

    #RADECOFFSET
    # x=np.arange
    # x=27.19*1e-3*np.arange(-512,512+1,1)

    plt.figure(figsize=(10,6))
    if vmin==None:
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(image, origin='lower',norm=norm)
    else:
        plt.imshow(image, origin='lower',vmin=vmin,vmax=vmax)#norm=norm)#norm)#vmin = np.median(data[data<0]),vmax=0.1*np.max(data))
    apertures_dao.plot(color='white', lw=1)
    # apertures_bpm.plot(color='red',lw=1)
    for i in range (len(detection_list['xcentroid'])):
        plt.text(detection_list['xcentroid'][i]+10, detection_list['ycentroid'][i]+5,str(i+1),color='white',weight='bold',fontsize=6)
    plt.title(source_name+": "+str(len(detection_list['xcentroid'])-1)+" local peaks above threshold \n (background+$5\sigma$) \n DAOFind Method")

    if extra_sources!=None:
#         for i in range (len(extra_sources['x_mean'])):
        positions_fake = np.transpose((extra_sources['xcentroid'], extra_sources['ycentroid']))
        apertures_fake = CircularAperture(positions_fake, r=5.)
        apertures_fake.plot(color='red')#, lw=1)
#             plt.text(extra_sources['x_mean'][i]+10, extra_sources['y_mean'][i]+5,str(i+1),color='red',weight='bold',fontsize=6)

    plt.show()

    return

#BASICPHOTOMETRY METHOD
def basicphotometry_method(image,detected_sources):
    sigma_psf=3.0
    daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()

    if 'xcentroid' not in detected_sources.colnames:
        detected_sources.rename_column('x_mean','xcentroid')
        detected_sources.rename_column('y_mean','ycentroid')

    # index_x=find_nearest(detected_sources['xcentroid'], value=512)
    # index_y=find_nearest(detected_sources['ycentroid'], value=512)
    # index=index_x
    if 'amplitude' not in detected_sources.keys():
        parameter=str('peak')
    else:
        parameter=str('amplitude')
    index=find_nearest(detected_sources[parameter], value=np.max(detected_sources[parameter]))
    x0,y0=detected_sources['xcentroid'][index],detected_sources['ycentroid'][index]
    # print("x0:",x0)
    # print("y0:", y0)
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)#, x_0=x0, y_0=y0, flux=sources['flux'][index])

    psf_model.x_0.fixed = True          #because the centroid positions are knwon
    psf_model.y_0.fixed = True
    # psf_model.sigma.fixed = False

    #pos=Table(names=['x_0', 'y_0'], data=[[512],[512]])
    pos = Table(names=['x_0', 'y_0'], data=[detected_sources['xcentroid'],detected_sources['ycentroid']])
    # pos = Table(names=['x_0', 'y_0'], data=[512,(512)])
    # print(pos)
    # print(psf_model)


    photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg,psf_model=psf_model,
                                    fitter=LevMarLSQFitter(),fitshape=(11,11))#,aperture_radius=8.)

    result_tab = photometry(image=image, init_guesses=pos)

    # get residual image
    residual_image = photometry.get_residual_image()

    for column in result_tab.columns:
        result_tab[column].info.format = '%.5g'


    return result_tab, residual_image, photometry, psf_model


def companion_parameters(psf_result,phys_distance,Lmag,flux_primary):
    Companions=Table(names=['Detection', 'x','y', 'Sep (as)','Sep (au)','flux_fit','flux_ratio','Lmag', 'deltaMag'])
    for col in Companions.colnames:
        Companions[col].info.format = '%.4g'
    # Companions['Lmag'].info.format = '%.2g'
    # Companions['deltaMag'].info.format = '%.2g'
    # Companions['Sep (au)'].info.format = '%.5g'
    # Companions['Sep (as)'].info.format = '%.3g'
    # Companions['flux_fit'].info.format = '%.3g'

    positions = np.transpose((psf_result['x_0'], psf_result['y_0']))
    apertures = CircularAperture(positions, r=8.)
    phys_distance=phys_distance     #0.6    #in parsec
    Lmag=Lmag   #3.8

    # plt.figure(figsize=(8, 6))
    # plt.imshow(data,origin='lower',vmin = np.median(data[data<0]),vmax=1*np.max(data))
    # apertures.plot(color='white', lw=1.5)
    # index=find_nearest(psf_result['x_0'], value=509)
    index_x=find_nearest(psf_result['flux_fit'], value=np.max(psf_result['flux_fit']))
    index_y=find_nearest(psf_result['flux_fit'], value=np.max(psf_result['flux_fit']))
    # print(index_x,index_y)
    if psf_result['flux_fit'][index_x]==np.max(psf_result['flux_fit']):
        index=index_x
    else:
        index=index_y

    # print(index)

    central_pixel_x,central_pixel_y=psf_result['x_0'][index],psf_result['y_0'][index]

    if flux_primary==None:
        flux_primary=psf_result['flux_fit'][index]

    for i in range (len(psf_result['x_0'])):
        distance_pixels=np.sqrt(np.abs(central_pixel_x-psf_result['x_0'][i])**2+np.abs(central_pixel_y-psf_result['y_0'][i])**2)
        distance_as=distance_pixels*27.053*1e-3         #27.053=pixelscale 25=7.053mas/pixel > returns value in arcsec
        distance_au=distance_as*phys_distance*1e3       #returns value in au
        flux_comp=psf_result['flux_fit'][i]
        flux_ratio=flux_comp/flux_primary
        mag_comp,delta_mag= magnitude_peak(Lmag,flux_primary,flux_comp)
        Companions.add_row([i+1,psf_result['x_0'][i],psf_result['y_0'][i],distance_as,distance_au,flux_comp,flux_ratio,mag_comp,delta_mag])
    #     plt.text(result_tab['x_0'][i]+10, result_tab['y_0'][i]+10,str(i+1),color='white',weight='bold')

    #print(Companions)

    return Companions,index,flux_primary
