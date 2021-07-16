import micasense.plotutils as plotutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os,glob
import math

import os, glob
from micasense.image import Image
from micasense.panel import Panel
import micasense.dls as dls
import micasense.capture as capture

import micasense.metadata as metadata
import micasense.utils as msutils
from skimage import measure

from osgeo import gdal, osr
import subprocess

class RadioCalibrate:
    def __init__(self):
        self.cap = None
        self.panelCap = None
        self.irr_from_panel = []
        self.irr_from_dls = []
        self.dls_correction = []
        
    def loadImage(self, imageNames):
        self.imageNames = imageNames
        imageNames_list = []
        for i in range(1,6):
            imageNames_list.append(imageNames + str(i) + '.tif')
        self.cap = capture.Capture.from_filelist(imageNames_list)
        
    def loadPanel(self, imageNames):
        self.panelCap = capture.Capture.from_filelist(imageNames)
        
    def computeRa2Re(self):
        self.irr_from_panel = []
        for k in range(5):
            img = self.panelCap.images[k]
            region = np.asarray(self.panelCap.panelCorners[k])

            markedImg = img.raw().copy()
            cv2.drawContours(markedImg,[region], 0, (0,0, 255), 3)
            plotutils.plotwithcolorbar(markedImg, 'Panel region in radiance image')

            rev_panel_pts = np.fliplr(region) #skimage and opencv coords are reversed
            w, h = img.raw().shape
            mask = measure.grid_points_in_poly((w,h),rev_panel_pts)
            num_pixels = mask.sum()
            panelRegion = img.radiance()[mask]
            meanRadiance = panelRegion.mean()
            
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]

            panelReflectance = panel_reflectance_by_band[k]
            self.irr_from_panel.append(meanRadiance * math.pi /panelReflectance )
            
        # Define DLS sensor orientation vector relative to dls pose frame
        dls_orientation_vector = np.array([0,0,-1])
        # compute sun orientation and sun-sensor angles
        (
            sun_vector_ned,    # Solar vector in North-East-Down coordinates
            sensor_vector_ned, # DLS vector in North-East-Down coordinates
            sun_sensor_angle,  # Angle between DLS vector and sun vector
            solar_elevation,   # Elevation of the sun above the horizon
            solar_azimuth,     # Azimuth (heading) of the sun
        ) = dls.compute_sun_angle(self.panelCap.location(),
                              self.panelCap.dls_pose(),
                              self.panelCap.utc_time(),
                              dls_orientation_vector)
        # Since the diffuser reflects more light at shallow angles than at steep angles,
        # we compute a correction for this
        fresnel_correction = dls.fresnel(sun_sensor_angle)

        # Now we can correct the raw DLS readings and compute the irradiance on level ground
        irr_from_dls = []
        for img in self.panelCap.images:
            dir_dif_ratio = 6.0
            percent_diffuse = 1.0/dir_dif_ratio
            # measured Irradiance / fresnelCorrection
            sensor_irradiance = img.spectral_irradiance / fresnel_correction
            untilted_direct_irr = sensor_irradiance / (percent_diffuse + np.cos(sun_sensor_angle))
            # compute irradiance on the ground using the solar altitude angle
            dls_irr = untilted_direct_irr * (percent_diffuse + np.sin(solar_elevation))
            irr_from_dls.append(dls_irr)
            
        self.dls_correction = []
        for k in range(5):
            self.dls_correction.append(self.irr_from_panel[k] / irr_from_dls[k])
            
    def compute_dls_irr(self):
        # Define DLS sensor orientation vector relative to dls pose frame
        dls_orientation_vector = np.array([0,0,-1])
        # compute sun orientation and sun-sensor angles
        (
            sun_vector_ned,    # Solar vector in North-East-Down coordinates
            sensor_vector_ned, # DLS vector in North-East-Down coordinates
            sun_sensor_angle,  # Angle between DLS vector and sun vector
            solar_elevation,   # Elevation of the sun above the horizon
            solar_azimuth,     # Azimuth (heading) of the sun
        ) = dls.compute_sun_angle(self.cap.location(),
                              self.cap.dls_pose(),
                              self.cap.utc_time(),
                              dls_orientation_vector)
        # Since the diffuser reflects more light at shallow angles than at steep angles,
        # we compute a correction for this
        fresnel_correction = dls.fresnel(sun_sensor_angle)

        # Now we can correct the raw DLS readings and compute the irradiance on level ground
        self.irr_from_dls = []
        for img in self.cap.images:
            dir_dif_ratio = 6.0
            percent_diffuse = 1.0/dir_dif_ratio
            # measured Irradiance / fresnelCorrection
            sensor_irradiance = img.spectral_irradiance / fresnel_correction
            untilted_direct_irr = sensor_irradiance / (percent_diffuse + np.cos(sun_sensor_angle))
            # compute irradiance on the ground using the solar altitude angle
            dls_irr = untilted_direct_irr * (percent_diffuse + np.sin(solar_elevation))
            self.irr_from_dls.append(dls_irr)
            
    def save_undistorted_reflectance(self, outName):
        bands = [1, 2, 3, 4, 5]
        for band in bands:
            band_reflectance = self.cap.images[band-1].undistorted_reflectance(self.irr_from_panel[band-1])
            
            nrows, ncols = band_reflectance.shape
            # Write output
            driver = gdal.GetDriverByName('Gtiff')
            Newdataset = driver.Create(outName+str(band)+'.tif', ncols, nrows, 1, gdal.GDT_UInt16)

            band_reflectance = band_reflectance * 65536
            band_reflectance[band_reflectance>=65536] = 65535
            band_reflectance = np.asarray(band_reflectance, np.uint16)
            
            Newdataset.GetRasterBand(1).WriteArray(band_reflectance)
            Newdataset = None
            
            
            srcImage = self.imageNames +str(band)+'.tif'
            dstImage = outName+str(band)+'.tif'
            subprocess.Popen(["exiftool","-tagsFromFile" , srcImage, "-all:all", "-xmp",
                             "-r", "-overwrite_original", dstImage])
            
    def save_dls_reflectance(self, outName):
        bands = [1, 2, 3, 4, 5]
        for band in bands:
            irr_correct = self.irr_from_dls[band-1] * self.dls_correction[band-1]
            band_reflectance = self.cap.images[band-1].undistorted_reflectance(irr_correct)
            
            nrows, ncols = band_reflectance.shape
            # Write output
            driver = gdal.GetDriverByName('Gtiff')
            Newdataset = driver.Create(outName+str(band)+'.tif', ncols, nrows, 1, gdal.GDT_UInt16)
            
            band_reflectance = band_reflectance * 65536
            band_reflectance[band_reflectance>=65536] = 65535
            band_reflectance = np.asarray(band_reflectance, np.uint16)
                       
            Newdataset.GetRasterBand(1).WriteArray(band_reflectance)
            Newdataset = None
            
            
            srcImage = self.imageNames +str(band)+'.tif'
            dstImage = outName+str(band)+'.tif'
            subprocess.Popen(["exiftool","-tagsFromFile" , srcImage, "-all:all", "-xmp",
                             "-r", "-overwrite_original", dstImage])

        
    
        
    def save_undistorted_radiance(self, outName):
        bands = [1, 2, 3, 4, 5]
        for band in bands:
            band_radiance = self.cap.images[band-1].undistorted_radiance()
            
            nrows, ncols = band_radiance.shape
            # Write output
            driver = gdal.GetDriverByName('Gtiff')
            Newdataset = driver.Create(outName+str(band)+'.tif', ncols, nrows, 1, gdal.GDT_UInt16)

            band_reflectance = band_reflectance * 65536
            band_reflectance[band_reflectance>=65536] = 65535
            band_reflectance = np.asarray(band_reflectance, np.uint16)
            
            Newdataset.GetRasterBand(1).WriteArray(band_radiance)
            Newdataset = None
            
            
            srcImage = self.imageNames +str(band)+'.tif'
            dstImage = outName+str(band)+'.tif'
            subprocess.Popen(["exiftool","-tagsFromFile" , srcImage, "-all:all", "-xmp",
                             "-r", "-overwrite_original", dstImage])
