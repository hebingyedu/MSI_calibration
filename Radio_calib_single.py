# +
import micasense.plotutils as plotutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os,glob
import math

import os, glob
from micasense.image import Image
from micasense.panel import Panel

import micasense.metadata as metadata
import micasense.utils as msutils
from skimage import measure

from osgeo import gdal, osr
import subprocess


# -

class Radio_calib:
    def __init__(self):
        self.panelImg = None
        self.panel = None

        self.radianceToReflectance = None
        self.panelMeta = None
        
        self.imgMeta = None
        
        self.dataset = None
        self.imageName = None
        self.outImageName = None
        
        self.undistortedReflectance = None
        self.inputCorners = None
        
        
    def ReadPanel(self, imageName):
        img = Image(imageName)
        self.panel = Panel(img)
        dataset = gdal.Open(imageName, gdal.GA_ReadOnly)
        band = dataset.GetRasterBand(1)
        self.panelImg = band.ReadAsArray()
        
        exiftoolPath = None
        if os.name == 'nt':
            exiftoolPath = 'C:/exiftool/exiftool.exe'
        self.panelMeta = metadata.Metadata(imageName, exiftoolPath=exiftoolPath)
        
        if not self.panel.panel_detected():
            print("Panel Not Detected!")
            
    def set_panel_conner(self, corner):
            self.inputCorners = corner
        
    def PanelInfo(self):
        print("Detected panel serial: {}".format(self.panel.serial))
        mean, std, num, sat_count = self.panel.raw()
        print("Extracted Panel Statistics:")
        print("Mean: {}".format(mean))
        print("Standard Deviation: {}".format(std))
        print("Panel Pixel Count: {}".format(num))
        print("Saturated Pixel Count: {}".format(sat_count))
        
        
        markedImg = self.panelImg.copy()
        cv2.drawContours(markedImg,[self.panel.panel_corners()], 0, (0,0, 255), 3)
        plotutils.plotwithcolorbar(markedImg, 'Panel region in radiance image')
        
    

        
    def GetRad2Ref(self):
        exiftoolPath = None

        cameraMake = self.panelMeta.get_item('EXIF:Make')
        cameraModel = self.panelMeta.get_item('EXIF:Model')
        firmwareVersion = self.panelMeta.get_item('EXIF:Software')
        self.PanelBandName = self.panelMeta.get_item('XMP:BandName')
        print('{0} {1} firmware version: {2}'.format(cameraMake, 
                                                     cameraModel, 
                                                     firmwareVersion))


        radianceImage, L, V, R = msutils.raw_image_to_radiance(self.panelMeta, self.panelImg)

        # Our panel calibration by band (from MicaSense for our specific panel)
        panelCalibration = { 
            "Blue": 0.67, 
            "Green": 0.69, 
            "Red": 0.68, 
            "Red edge": 0.67, 
            "NIR": 0.61 
        }
        
        if not self.panel.panel_detected():
            region = self.inputCorners
            markedImg = self.panelImg.copy()
            cv2.drawContours(markedImg,[self.inputCorners], 0, (0,0, 255), 3)
            plotutils.plotwithcolorbar(markedImg, 'Panel region in radiance image')
        else:
            region = self.panel.panel_corners()
            markedImg = self.panelImg.copy()
            cv2.drawContours(markedImg,[self.panel.panel_corners()], 0, (0,0, 255), 3)
            plotutils.plotwithcolorbar(markedImg, 'Panel region in radiance image')
        rev_panel_pts = np.fliplr(region) #skimage and opencv coords are reversed
        w, h = radianceImage.shape
        mask = measure.grid_points_in_poly((w,h),rev_panel_pts)
        num_pixels = mask.sum()
        panelRegion = radianceImage[mask]
        meanRadiance = panelRegion.mean()

#         # Select panel region from radiance image
#         panelRegion = radianceImage[self.uly:self.lry, self.ulx:self.lrx]
#         meanRadiance = panelRegion.mean()
        # print('Mean Radiance in panel region: {:1.3f} W/m^2/nm/sr'.format(meanRadiance))
        panelReflectance = panelCalibration[self.PanelBandName]
        self.radianceToReflectance = panelReflectance / meanRadiance
        
    def calibration(self, imageName, isplot=False):
        self.imageName = imageName
        
        self.dataset = gdal.Open(imageName, gdal.GA_ReadOnly)
        band = self.dataset.GetRasterBand(1)
        imageRaw = band.ReadAsArray()

        exiftoolPath = None
        if os.name == 'nt':
            exiftoolPath = 'C:/exiftool/exiftool.exe'
        # get image metadata
        meta = metadata.Metadata(imageName, exiftoolPath=exiftoolPath)
        self.imgMeta = meta
        
        cameraMake = meta.get_item('EXIF:Make')
        cameraModel = meta.get_item('EXIF:Model')
        firmwareVersion = meta.get_item('EXIF:Software')
        bandName = meta.get_item('XMP:BandName')
#         print('{0} {1} firmware version: {2}'.format(cameraMake, 
#                                                      cameraModel, 
#                                                      firmwareVersion))
#         print('Exposure Time: {0} seconds'.format(meta.get_item('EXIF:ExposureTime')))
#         print('Imager Gain: {0}'.format(meta.get_item('EXIF:ISOSpeed')/100.0))
#         print('Size: {0}x{1} pixels'.format(meta.get_item('EXIF:ImageWidth'),meta.get_item('EXIF:ImageHeight')))
#         print('Band Name: {0}'.format(bandName))
#         print('Center Wavelength: {0} nm'.format(meta.get_item('XMP:CentralWavelength')))
#         print('Bandwidth: {0} nm'.format(meta.get_item('XMP:WavelengthFWHM')))
#         print('Capture ID: {0}'.format(meta.get_item('XMP:CaptureId')))
#         print('Flight ID: {0}'.format(meta.get_item('XMP:FlightId')))
#         print('Focal Length: {0}'.format(meta.get_item('XMP:FocalLength')))

        radianceImage, L, V, R = msutils.raw_image_to_radiance(meta, imageRaw)

        reflectanceImage = radianceImage * self.radianceToReflectance

        # correct for lens distortions to make straight lines straight
        self.undistortedReflectance = msutils.correct_lens_distortion(meta, reflectanceImage)
        
        if(isplot):
            fig = plotutils.plotwithcolorbar(imageRaw, title='Raw image values with colorbar')
            plotutils.plotwithcolorbar(radianceImage, 'Converted Reflectane Image');
            plotutils.plotwithcolorbar(self.undistortedReflectance, 'Undistorted reflectance image');
            
    def saveRef(self, outImageNage):
        self.outImageName = outImageNage
        
        calibratedimg = self.undistortedReflectance
        calibratedimg = np.asarray(calibratedimg*2**16, np.uint16)
        
        gt_ref = self.dataset.GetGeoTransform()
        prj_ref = self.dataset.GetProjection()
        srs = osr.SpatialReference(wkt=prj_ref)
        if srs.IsProjected:
            print(srs.GetAttrValue('projcs'))
        print(srs.GetAttrValue('geogcs'))

        # Make geotransform
        xmin, ymax = (gt_ref[0], gt_ref[3])
        nrows, ncols = calibratedimg.shape
        xres = gt_ref[1]
        yres = np.abs(gt_ref[5])
        geotransform = (xmin, xres, 0, ymax, 0, -yres)

        # Write output
        driver = gdal.GetDriverByName('Gtiff')
        Newdataset = driver.Create(outImageNage, ncols, nrows, 1, gdal.GDT_UInt16)
        Newdataset.SetGeoTransform(geotransform)
        srs = osr.SpatialReference(wkt=prj_ref)
        Newdataset.SetProjection(srs.ExportToWkt())
        Newdataset.GetRasterBand(1).WriteArray(calibratedimg)
        Newdataset = None

        subprocess.Popen(["exiftool","-tagsFromFile" , self.imageName, "-all:all>all:all", "-xmp",
                         "-r", "-overwrite_original", outImageNage])

