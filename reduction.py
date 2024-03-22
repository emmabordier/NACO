#%pylab inline
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
from photutils.psf import IterativelySubtractedPSFPhotometry
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def background_estimation(data):
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    return bkg

def list_of_files(dir_data):

    all_files=[file for file in sorted(glob(dir_data+"*")) if os.path.basename(file).startswith("NACO") and file.endswith("fits")]
    files=[file for file in all_files if fits.open(file)[0].header["HIERARCH ESO DPR CATG"]=="SCIENCE"]
    #print(*files,sep="\n")
    cut=int((len(files)-2)/2)
#     print(cut)
    obs_1=files[:cut+1]
    obs_2=files[cut+1:len(files)]
    # print(*obs_1,sep = "\n")
    # print("youhou")
    # print(*obs_2,sep = "\n")
    science=[file for file in files if fits.open(file)[0].header["HIERARCH ESO DPR TYPE"]=="OBJECT"]
    sky=[file for file in files if fits.open(file)[0].header["HIERARCH ESO DPR TYPE"]=="SKY"]

    return science, sky, obs_1, obs_2, cut

#sky_estimation from the science frames
def sky_estimations_science(science_files):
    sky_estimations_science={}
    for observation in range(len(science_files)):
        background_median=[]
        background_rms_median=[]
        data=fits.open(science_files[observation])[0].data
        for frame in range(np.size(data,axis=0)):
            data_frame=data[frame]
            bkg=background_estimation(data_frame)
            background_median.append(bkg.background_median)
            background_rms_median.append(bkg.background_rms_median)
        sky_estimations_science[observation]=background_median,background_rms_median

    return sky_estimations_science

#sky estimation from the sky frames
def sky_estimations_sky(sky_files):
    sky_estimations_sky={}
    for observation in range(len(sky_files)):
        background_median=[]
        background_rms_median=[]
        data=fits.open(sky_files[observation])[0].data
        for frame in range(np.size(data,axis=0)):
            data_frame=data[frame]
            bkg=background_estimation(data_frame)
            background_median.append(bkg.background_median)
            background_rms_median.append(bkg.background_rms_median)
        sky_estimations_sky[observation]=background_median,background_rms_median

    return sky_estimations_sky

def plot_frames_intensity(source_name,epoch,sky_est_science, sky_est_sky, cut, first_frame):
    dir_reduced='/Users/ebordier/NACO/REDUCED/'+source_name+'_2/'
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6))
    size_science=len(sky_est_science[0][1])
    size_sky=len(sky_est_sky[0][1])
    for i in range(len(sky_est_science.keys())):      #for each observations (around 8)
        if i<int(cut)/2:
            ax1.plot(sky_est_science[i][0][first_frame:size_science],label=i+1,linestyle="dotted")     #If I discard the first 2 frames
            ax2.plot(sky_est_science[i][1][first_frame:size_science],label=i+1,linestyle="dotted")
            ax1.plot(sky_est_sky[i][0][first_frame:size_sky],label=i+1,linestyle="dashed")  #If I discard the first 2 frames
            ax2.plot(sky_est_sky[i][1][first_frame:size_sky],label=i+1,linestyle="dashed")
    ax1.set_title("Background Median variation of CYCLE 1 for \n" + source_name + ", Date: "+  epoch + "\n" + "Dotted Line : SCIENCE, Dashed Line : SKY "  )
    ax2.set_title("Background Median RMS variation of CYCLE 1 for \n"+ source_name + ", Date: " + epoch + "\n" + "Dotted Line : SCIENCE, Dashed Line : SKY ")
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("Frame number")
    ax2.set_xlabel("Frame number")
    plt.savefig(dir_reduced+"/background_variations_cycle1.png")

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 6))
    for i in range(len(sky_est_sky.keys())):      #for each observations (around 8)
        if i>=int(cut)/2:
    #         if i==7 or i==6:
    #             continue
            ax1.plot(sky_est_science[i][0][first_frame:size_science],label=i+1,linestyle="dotted") #[5:size_science-1]
            ax2.plot(sky_est_science[i][1][first_frame:size_science],label=i+1,linestyle="dotted")
            ax1.plot(sky_est_sky[i][0][first_frame:size_sky],label=i+1,linestyle="dashed")#[5:size_sky-1]
            ax2.plot(sky_est_sky[i][1][first_frame:size_sky],label=i+1,linestyle="dashed")
    ax1.set_title("Background Median variation of CYCLE 2 for \n " + source_name + ", Date: "+ epoch + "\n" + "Dotted Line : SCIENCE, Dashed Line : SKY "  )
    ax2.set_title("Background Median RMS variation of CYCLE 2 for \n "+ source_name + ", Date: "+ epoch + "\n" + "Dotted Line : SCIENCE, Dashed Line : SKY ")
    ax1.legend()
    ax2.legend()
    plt.savefig(dir_reduced+"/background_variations_cycle2.png")
    plt.show()

    return

#REDUCTION ROUTINE

def reduction(obs_1,obs_2,first_frame_sky,last_frame_sky,first_frame_science,last_frame_science):
# master_sky=[]
    master_image=[]
    master_sky=[]
    for list_file in [obs_1,obs_2]:      #in order to reduce separately given that we have 2 cycles

        science=[file for file in list_file if fits.open(file)[0].header["HIERARCH ESO DPR TYPE"]=="OBJECT"]
        sky=[file for file in list_file if fits.open(file)[0].header["HIERARCH ESO DPR TYPE"]=="SKY"]

       #TO BE REMOVED IF NO PROBLEM WITH CYCLES
    #     if list_file in obs_2:
    #         science.pop(6,7)
    #     print(science)

        median_sky=[]
        for file in sky:
            data=fits.open(file)[0].data
            #if list_file='obs_1':
            median_sky.append(np.median(data[first_frame_sky:last_frame_sky],axis=0))
    #         else:
    #         median_sky.append(np.median(data[10:],axis=0))
    #     print(np.median(median_sky,axis=0))
    #     print(np.median(np.stack(median_sky,axis=0)))
        median_sky=np.median(median_sky,axis=0)     #median of the stack image
#         print(np.shape(median_sky))
        master_sky.append(median_sky)
        science_data=[]
        for file in science:
            data=fits.open(file)[0].data
            data=data-median_sky
            science_data.append(np.median(data[first_frame_science:last_frame_science],axis=0)) #median each cube over its 126 images to get 1 images
    #     science_data=np.median(science_data,axis=0)
#         print(np.shape(science_data))

        image=np.sum(science_data,axis=0)#np.sum(np.stack)
        master_image.append(image)
#         print(np.shape(master_image))
    #     master_image_2.append(science_data)#master_image.append(image)
    #     master_sky.append(median_sky)

    return master_image, master_sky        #master_image = reduced image for each cycle

def alignment(cut,cycle2,master_image):
    header=fits.open(cycle2[5])[0].header
    offsetx=int(-header['HIERARCH ESO SEQ CUMOFFSETY'])     #for G268.3957: offsety=-46 ET offsetx=167
    offsety=int(-header['HIERARCH ESO SEQ CUMOFFSETX'])

    (imx,imy)=master_image[1].shape      #images[1]= median second cycle
    image_2_shift=np.zeros_like(master_image[1])  #reconstructing image with the offset
    ox,oy=offsetx,offsety       #SEQ.CUMOFFSETX/Y
    non = lambda s: s if s<0 else None
    mom = lambda s: max(0,s)
    image_2_shift[mom(ox):non(ox),mom(oy):non(oy)]=master_image[1][mom(-ox):non(-ox), mom(-oy):non(-oy)]

    stack=[]
    stack.append(master_image[0])      #cycle1
    stack.append(image_2_shift)  #cycle2
    master=np.sum(stack,axis=0)

    return master, offsetx, offsety
