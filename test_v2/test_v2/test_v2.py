#from pip._internal import main as pipmain
#pipmain(['install','scipy'])

import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import math
import ctypes
import numpy.ctypeslib as ctl
from scipy import ndimage
#
# test_v2
#
#11
class test_v2(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "test_v2" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# test_v2Widget
#

class test_v2Widget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  #
  # Load C functions
  #
  priorlib=ctl.ctypes_load_library('libsegment.so', '/Users/junyuchen/Desktop/slicerPython/test_v2/test_v2/')
  # MRF_EM
  c_func_MRF_EM = priorlib.MRF_EM
  c_func_MRF_EM.restype=ctypes.c_int
  c_func_MRF_EM.argtypes= \
	  [ctl.ndpointer(np.float32,flags='c_contiguous'), #voxels
	  ctl.ndpointer(np.int16, flags='c_contiguous'), #output labels
	  ctypes.c_int, #xdim
	  ctypes.c_int, #ydim
	  ctypes.c_int, #zdim
	  ctypes.c_int, #nclasses
	  ctypes.c_double, # beta
    ]

    # MRF_EM_ctinfo
  c_func_MRF_EM_ctinfo = priorlib.MRF_EM_ctinfo
  c_func_MRF_EM_ctinfo.restype=ctypes.c_int
  c_func_MRF_EM_ctinfo.argtypes= \
	  [ctl.ndpointer(np.float32, flags='c_contiguous'), #voxels
	  ctl.ndpointer(np.int16, flags='c_contiguous'), #output labels
    ctl.ndpointer(np.int16, flags='c_contiguous'), #CT prior labels
	  ctypes.c_int, #xdim
	  ctypes.c_int, #ydim
	  ctypes.c_int, #zdim
	  ctypes.c_int, #nclasses
	  ctypes.c_double, # beta
    ctypes.c_double, # gamma
    ]

  # region growing
  c_func_region_grow_cropped3Dimg = priorlib.region_grow_cropped3Dimg
  c_func_region_grow_cropped3Dimg.restype=ctypes.c_int
  c_func_region_grow_cropped3Dimg.argtypes= \
	  [ctl.ndpointer(np.int16, flags='c_contiguous'), #input labels
	  ctypes.c_int, #xdim
	  ctypes.c_int, #ydim
	  ctypes.c_int, #zdim
	  ctypes.c_double, # seedx
    ctypes.c_double, # seedy
    ctypes.c_double, # seedz
    ctypes.c_int, #nclasses
    ]
  def setup(self):

    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Inputs"
    self.layout.addWidget(inputCollapsibleButton)

    outputsCollapsibleButton = ctk.ctkCollapsibleButton()
    outputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(outputsCollapsibleButton)

    # Layout within the dummy collapsible button
    inputsFormLayout = qt.QFormLayout(inputCollapsibleButton)
    outputsFormLayout = qt.QFormLayout(outputsCollapsibleButton)
    
    #
    # input volume selector SPECT
    #
    self.inputSelector_spect = slicer.qMRMLNodeComboBox()
    self.inputSelector_spect.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector_spect.selectNodeUponCreation = True
    self.inputSelector_spect.addEnabled = False
    self.inputSelector_spect.removeEnabled = False
    self.inputSelector_spect.noneEnabled = False
    self.inputSelector_spect.showHidden = False
    self.inputSelector_spect.showChildNodeTypes = False
    self.inputSelector_spect.setMRMLScene( slicer.mrmlScene )
    self.inputSelector_spect.setToolTip( "Pick SPECT image" )
    inputsFormLayout.addRow("SPECT Volume: ", self.inputSelector_spect)

    #
    # input volume selector
    #
    self.inputSelector_ct = slicer.qMRMLNodeComboBox()
    self.inputSelector_ct.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector_ct.selectNodeUponCreation = True
    self.inputSelector_ct.addEnabled = False
    self.inputSelector_ct.removeEnabled = False
    self.inputSelector_ct.noneEnabled = False
    self.inputSelector_ct.showHidden = False
    self.inputSelector_ct.showChildNodeTypes = False
    self.inputSelector_ct.setMRMLScene( slicer.mrmlScene )
    self.inputSelector_ct.setToolTip( "Pick CT image" )
    inputsFormLayout.addRow("CT Volume: ", self.inputSelector_ct)

    #
    # output clustering volume selector
    #
    self.outputLesClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputLesClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputLesClusterSelector.selectNodeUponCreation = True
    self.outputLesClusterSelector.addEnabled = True
    self.outputLesClusterSelector.removeEnabled = True
    self.outputLesClusterSelector.noneEnabled = True
    self.outputLesClusterSelector.showHidden = False
    self.outputLesClusterSelector.showChildNodeTypes = False
    self.outputLesClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputLesClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lesion Label Map: ", self.outputLesClusterSelector)

    #
    # output volume selector
    #
    self.outputBoneClusterSelector = slicer.qMRMLNodeComboBox()
    self.outputBoneClusterSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.outputBoneClusterSelector.selectNodeUponCreation = True
    self.outputBoneClusterSelector.addEnabled = True
    self.outputBoneClusterSelector.removeEnabled = True
    self.outputBoneClusterSelector.noneEnabled = True
    self.outputBoneClusterSelector.showHidden = False
    self.outputBoneClusterSelector.showChildNodeTypes = False
    self.outputBoneClusterSelector.setMRMLScene( slicer.mrmlScene )
    self.outputBoneClusterSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Bone Label Map: ", self.outputBoneClusterSelector)

    #
    # output Les seg selector
    #
    self.segLesSelector = slicer.qMRMLNodeComboBox()
    self.segLesSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segLesSelector.selectNodeUponCreation = True
    self.segLesSelector.addEnabled = True
    self.segLesSelector.removeEnabled = True
    self.segLesSelector.noneEnabled = True
    self.segLesSelector.showHidden = False
    self.segLesSelector.showChildNodeTypes = False
    self.segLesSelector.setMRMLScene( slicer.mrmlScene )
    self.segLesSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Lesion Segmentation: ", self.segLesSelector)

    #
    # output Bone seg selector
    #
    self.segBoneSelector = slicer.qMRMLNodeComboBox()
    self.segBoneSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segBoneSelector.selectNodeUponCreation = True
    self.segBoneSelector.addEnabled = True
    self.segBoneSelector.removeEnabled = True
    self.segBoneSelector.noneEnabled = True
    self.segBoneSelector.showHidden = False
    self.segBoneSelector.showChildNodeTypes = False
    self.segBoneSelector.setMRMLScene( slicer.mrmlScene )
    self.segBoneSelector.setToolTip( "Pick the output to the algorithm." )
    outputsFormLayout.addRow("Output Bone Segmentation: ", self.segBoneSelector)

    #
    # cort bone seed point selector
    #
    self.cortboneseedSelector = slicer.qMRMLNodeComboBox()
    self.cortboneseedSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.cortboneseedSelector.selectNodeUponCreation = True
    self.cortboneseedSelector.addEnabled = True
    self.cortboneseedSelector.removeEnabled = True
    self.cortboneseedSelector.noneEnabled = True
    self.cortboneseedSelector.showHidden = False
    self.cortboneseedSelector.showChildNodeTypes = False
    self.cortboneseedSelector.setMRMLScene( slicer.mrmlScene )
    self.cortboneseedSelector.setToolTip( "Pick the output to the algorithm." )
    inputsFormLayout.addRow("Cort. Bone Seed in CT: ", self.cortboneseedSelector)

    #
    # trab bone seed point selector
    #
    self.trabboneseedSelector = slicer.qMRMLNodeComboBox()
    self.trabboneseedSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.trabboneseedSelector.selectNodeUponCreation = True
    self.trabboneseedSelector.addEnabled = True
    self.trabboneseedSelector.removeEnabled = True
    self.trabboneseedSelector.noneEnabled = True
    self.trabboneseedSelector.showHidden = False
    self.trabboneseedSelector.showChildNodeTypes = False
    self.trabboneseedSelector.setMRMLScene( slicer.mrmlScene )
    self.trabboneseedSelector.setToolTip( "Pick the output to the algorithm." )
    inputsFormLayout.addRow("Trab. Bone Seed in CT: ", self.trabboneseedSelector)


    #
    # lesion seed point selector
    #
    self.seedSelector = slicer.qMRMLNodeComboBox()
    self.seedSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.seedSelector.selectNodeUponCreation = True
    self.seedSelector.addEnabled = True
    self.seedSelector.removeEnabled = True
    self.seedSelector.noneEnabled = True
    self.seedSelector.showHidden = False
    self.seedSelector.showChildNodeTypes = False
    self.seedSelector.setMRMLScene( slicer.mrmlScene )
    self.seedSelector.setToolTip( "Pick the output to the algorithm." )
    inputsFormLayout.addRow("Lesion Seed in SPECT: ", self.seedSelector)

    #
    # bone seed point selector
    #
    self.boneseedSelector = slicer.qMRMLNodeComboBox()
    self.boneseedSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.boneseedSelector.selectNodeUponCreation = True
    self.boneseedSelector.addEnabled = True
    self.boneseedSelector.removeEnabled = True
    self.boneseedSelector.noneEnabled = True
    self.boneseedSelector.showHidden = False
    self.boneseedSelector.showChildNodeTypes = False
    self.boneseedSelector.setMRMLScene( slicer.mrmlScene )
    self.boneseedSelector.setToolTip( "Pick the output to the algorithm." )
    inputsFormLayout.addRow("Bone Seed in SPECT: ", self.boneseedSelector)


    #
    # seed point selector
    #
    self.roiSelector = slicer.qMRMLNodeComboBox()
    self.roiSelector.nodeTypes = ["vtkMRMLAnnotationROINode"]
    self.roiSelector.selectNodeUponCreation = True
    self.roiSelector.addEnabled = True
    self.roiSelector.removeEnabled = True
    self.roiSelector.noneEnabled = True
    self.roiSelector.showHidden = False
    self.roiSelector.showChildNodeTypes = False
    self.roiSelector.setMRMLScene( slicer.mrmlScene )
    self.roiSelector.setToolTip( "Pick the output to the algorithm." )
    inputsFormLayout.addRow("3D ROI: ", self.roiSelector)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    inputsFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    self.layout.addWidget(self.applyButton)

    #
    # Advanced Button
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced"
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
    #
    # beta value
    #
    self.betaSliderWidget = ctk.ctkSliderWidget()
    self.betaSliderWidget.singleStep = 0.01
    self.betaSliderWidget.minimum = 0
    self.betaSliderWidget.maximum = 100
    self.betaSliderWidget.value = 0.4
    self.betaSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    advancedFormLayout.addRow("Beta: ", self.betaSliderWidget)

    #
    # delta value
    #
    self.deltaSliderWidget = ctk.ctkSliderWidget()
    self.deltaSliderWidget.singleStep = 0.01
    self.deltaSliderWidget.minimum = 0
    self.deltaSliderWidget.maximum = 100
    self.deltaSliderWidget.value = 0.1
    self.deltaSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    advancedFormLayout.addRow("Gamma: ", self.deltaSliderWidget)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector_spect.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelector_ct.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputBoneClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputLesClusterSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segLesSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.segBoneSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.cortboneseedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.trabboneseedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.seedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.boneseedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector_spect.currentNode() and self.inputSelector_ct.currentNode() and self.outputBoneClusterSelector.currentNode() and self.seedSelector.currentNode() and self.boneseedSelector.currentNode() and self.cortboneseedSelector.currentNode() and self.trabboneseedSelector.currentNode() and self.roiSelector.currentNode() and self.segLesSelector.currentNode() and self.segBoneSelector.currentNode() and self.outputLesClusterSelector.currentNode()

  def onApplyButton(self):
    logic = test_v2Logic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    beta = self.betaSliderWidget.value
    delta = self.deltaSliderWidget.value
    logic.run(self.inputSelector_spect.currentNode(), self.inputSelector_ct.currentNode(), self.outputBoneClusterSelector.currentNode(), self.outputLesClusterSelector.currentNode(), self.segLesSelector.currentNode(), self.segBoneSelector.currentNode(), self.cortboneseedSelector.currentNode(), self.trabboneseedSelector.currentNode(), self.boneseedSelector.currentNode(), self.seedSelector.currentNode(), self.roiSelector.currentNode(), beta, delta, enableScreenshotsFlag)

#
# testLogic
#



#
# test_v2Logic
#

class test_v2Logic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode_spect, inputVolumeNode_ct , outputBoneClusterNode, outputLesClusterNode, outputLesSegmentationNode, outputBoneSegmentationNode, cortseedsNode, trabseedsNode, boneseedsNode, lesionseedsNode, roi3DNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode_spect:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not inputVolumeNode_ct:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputBoneClusterNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if not outputLesSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputBoneSegmentationNode:
      logging.debug('isValidInputOutputData failed: no output segmentation node defined')
      return False
    if not outputLesClusterNode:
      logging.debug('isValidInputOutputData failed: no output cluster node defined')
      return False
    if not cortseedsNode:
      logging.debug('isValidInputOutputData failed: no bone seeds defined')
      return False
    if not trabseedsNode:
      logging.debug('isValidInputOutputData failed: no bone seeds defined')
      return False
    if not lesionseedsNode:
      logging.debug('isValidInputOutputData failed: no seeds defined')
      return False
    if not boneseedsNode:
      logging.debug('isValidInputOutputData failed: no bone seeds defined')
      return False
    if not roi3DNode:
      logging.debug('isValidInputOutputData failed: no ROI defined')
      return False
    #if inputVolumeNode_spect.GetID()==outputLesVolumeNode.GetID():
     # logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      #return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)
    

  def extract_reg_w_same_label(self, label_map_in, seedx, seedy, seedz):
    label_map = np.zeros(label_map_in.shape)
    #print(label_map_in.shape)
    #print([seedx, seedy, seedz])
    label_map[label_map_in == label_map_in[seedx, seedy, seedz]] = 1
    return label_map

  
  def seg_alg_CT(self, img_3D_ct, roi_3D_cord_CT, beta):
    im_roi_3D_ct = img_3D_ct[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]]
    
    zdim,ydim,xdim = im_roi_3D_ct.shape

    CT_labeled_roi  = np.zeros(np.shape(im_roi_3D_ct))
    
    outputLabel = np.zeros(np.shape(img_3D_ct))
    num_clus_ct = 4

    ################ cluster and region growing on CT images ###################
    CT_labeled_roi  = np.ascontiguousarray(CT_labeled_roi, np.int16)
    im_roi_3D_ct = np.ascontiguousarray(im_roi_3D_ct, np.float32)
    test_v2Widget.c_func_MRF_EM(im_roi_3D_ct, CT_labeled_roi, xdim, ydim, zdim, num_clus_ct, float(beta))
    print(CT_labeled_roi)
    outputLabel[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]] = CT_labeled_roi
    return outputLabel

  def seg_alg_SPECT(self,img_3D_spect, img_3D_ct, CT_labeled, roi_3D_cord_SPECT, roi_3D_cord_CT, beta, gamma):
    im_roi_3D_spect = img_3D_spect[roi_3D_cord_SPECT[0]:roi_3D_cord_SPECT[1],roi_3D_cord_SPECT[2]:roi_3D_cord_SPECT[3],roi_3D_cord_SPECT[4]:roi_3D_cord_SPECT[5]]
    CT_labeled_roi = CT_labeled[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]]
    ### downsample
    x_spect,y_spect,z_spect = im_roi_3D_spect.shape
    x_ct,y_ct,z_ct = CT_labeled_roi.shape

    dsfactor = [x_spect/float(x_ct),y_spect/float(y_ct),z_spect/float(z_ct)]
    CT_labeled_roi = ndimage.interpolation.zoom(CT_labeled_roi, zoom=dsfactor,order=0)
    CT_labeled_roi = np.ascontiguousarray(CT_labeled_roi, np.int16)

    print('CT_size')
    print(CT_labeled_roi.shape)

    print('im_roi_3D_spect size')
    print(im_roi_3D_spect.shape)
    
    ############## Resize CT image ##################################
    zdim,ydim,xdim = im_roi_3D_spect.shape

    SPECT_labeled_roi  = np.zeros(np.shape(im_roi_3D_spect))
    
    print('gamma: ' + str(gamma))
    outputLabel = np.zeros(np.shape(img_3D_ct))
    num_clus_spect = 3

    ################ clustering on SPECT images ###################
    SPECT_labeled_roi  = np.ascontiguousarray(SPECT_labeled_roi, np.int16)
    im_roi_3D_spect = np.ascontiguousarray(im_roi_3D_spect, np.float32)
    test_v2Widget.c_func_MRF_EM_ctinfo(im_roi_3D_spect, SPECT_labeled_roi, CT_labeled_roi, xdim, ydim, zdim, num_clus_spect, float(beta), float(gamma))
    dsfactor = [x_ct/float(x_spect),y_ct/float(y_spect),z_ct/float(z_spect)]
    SPECT_labeled_roi = ndimage.interpolation.zoom(SPECT_labeled_roi, zoom=dsfactor,order=0)
    outputLabel[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]] = SPECT_labeled_roi
    return outputLabel

  def run(self, inputVolume_spect, inputVolume_ct,outputBoneClusterVolume, outputLesClusterVolume,outputLesSegmentation, outputBoneSegmentation, cortbone_seeds, trabbone_seeds, bone_seeds, les_seeds, roi3D, beta, gamma, enableScreenshots=0):
    
    
    """
    Run the actual algorithm
    """
    outputLesSegmentation.GetSegmentation().RemoveAllSegments()
    outputBoneSegmentation.GetSegmentation().RemoveAllSegments()
    if not self.isValidInputOutputData(inputVolume_spect, inputVolume_ct, outputBoneClusterVolume, outputLesClusterVolume, outputLesSegmentation, outputBoneSegmentation, cortbone_seeds, trabbone_seeds, bone_seeds, les_seeds, roi3D):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

     # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputBoneClusterVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    cliParams = {'InputVolume': inputVolume_ct.GetID(), 'OutputVolume': outputLesClusterVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    #cliParams = {'inputVolume_spect': inputVolume_spect.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    #cliParams = {'inputVolume_ct': inputVolume_ct.GetID(), 'OutputVolume': outputClusterVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    #cliParams = {'inputVolume_ct': inputVolume_ct.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    #cliParams = {'inputVolume_spect': inputVolume_spect.GetID(), 'OutputVolume': outputClusterVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)
    #cliParams = {'inputVolume_spect': inputVolume_ct.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : beta, 'ThresholdType' : 'Above'}
    #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('testTest-Start','MyScreenshot',-1)


    # convert volume to nd array
    spect_img = list(slicer.util.arrayFromVolume(inputVolume_spect))
    spect_img = np.asarray(spect_img)

    # convert volume to nd array
    ct_img = list(slicer.util.arrayFromVolume(inputVolume_ct))
    ct_img = np.asarray(ct_img)
    
    vol_size = inputVolume_spect.GetImageData().GetDimensions()
    vol_size = np.asarray(vol_size)
    vol_center = vol_size/2

    print('dimension is: ' + str(inputVolume_spect.GetImageData().GetDimensions()))
    print('spect size: '+str(np.shape(spect_img)))
    print('ct size: '+str(np.shape(ct_img)))
    ############################ ROI SPECT #########################################
    bounds = [0.0,0.0,0.0,0.0,0.0,0.0]
    roi3D.GetRASBounds(bounds)
    roiCenter = [0.0,0.0,0.0]
    roi3D.GetXYZ(roiCenter)
    roiCenter.append(1.0)
    # change roi coordinates to ijk
    print('roi center is: '+str(roiCenter))
    print('roi bound is: '+str(bounds))
    print('center is: '+str(vol_center))

    bounds_max = [0.,0.,0.,0.]; bounds_min = [0.,0.,0.,0.]
    bounds_max[0] = bounds[0]; bounds_max[1] = bounds[2]; bounds_max[2] = bounds[4]; bounds_max[3] = 1.0
    bounds_min[0] = bounds[1]; bounds_min[1] = bounds[3]; bounds_min[2] = bounds[5]; bounds_min[3] = 1.0
    
    b_max = vtk.vtkMatrix4x4()
    inputVolume_spect.GetRASToIJKMatrix(b_max)
    bounds_max_ijk = b_max.MultiplyDoublePoint(bounds_max)

    b_min = vtk.vtkMatrix4x4()
    inputVolume_spect.GetRASToIJKMatrix(b_min)
    bounds_min_ijk = b_min.MultiplyDoublePoint(bounds_min)

    roi_c = vtk.vtkMatrix4x4()
    inputVolume_spect.GetRASToIJKMatrix(roi_c)
    roiCenter_ijk = b_min.MultiplyDoublePoint(roiCenter)

    roiCenter_ijk = np.asarray(roiCenter_ijk)
    bounds_min_ijk = np.asarray(bounds_min_ijk)
    bounds_max_ijk = np.asarray(bounds_max_ijk)

    roi_x = [int(bounds_max_ijk[2]),int(bounds_min_ijk[2])]
    roi_y = [int(bounds_min_ijk[1]),int(bounds_max_ijk[1])]
    roi_z = [int(bounds_min_ijk[0]),int(bounds_max_ijk[0])]
    roi_3D_cord_SPECT = [np.min(roi_x),np.max(roi_x),np.min(roi_y),np.max(roi_y),np.min(roi_z),np.max(roi_z)]
    roi_3D_cord_SPECT = np.add(roi_3D_cord_SPECT, 1)

    #print('roi center-ijk is: '+str(np.round(roiCenter_ijk)))
    #print('roi bound min-ijk is: '+str(np.round(bounds_min_ijk)))
    #print('roi bound max-ijk is: '+str(np.round(bounds_max_ijk)))

    ############################ ROI CT #########################################
    bounds = [0.0,0.0,0.0,0.0,0.0,0.0]
    roi3D.GetRASBounds(bounds)
    roiCenter = [0.0,0.0,0.0]
    roi3D.GetXYZ(roiCenter)
    roiCenter.append(1.0)
    # change roi coordinates to ijk
    print('roi center is: '+str(roiCenter))
    print('roi bound is: '+str(bounds))
    print('center is: '+str(vol_center))

    bounds_max = [0.,0.,0.,0.]; bounds_min = [0.,0.,0.,0.]
    bounds_max[0] = bounds[0]; bounds_max[1] = bounds[2]; bounds_max[2] = bounds[4]; bounds_max[3] = 1.0
    bounds_min[0] = bounds[1]; bounds_min[1] = bounds[3]; bounds_min[2] = bounds[5]; bounds_min[3] = 1.0
    
    b_max = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(b_max)
    bounds_max_ijk = b_max.MultiplyDoublePoint(bounds_max)

    b_min = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(b_min)
    bounds_min_ijk = b_min.MultiplyDoublePoint(bounds_min)

    roi_c = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(roi_c)
    roiCenter_ijk = b_min.MultiplyDoublePoint(roiCenter)

    roiCenter_ijk = np.asarray(roiCenter_ijk)
    bounds_min_ijk = np.asarray(bounds_min_ijk)
    bounds_max_ijk = np.asarray(bounds_max_ijk)

    roi_x = [int(bounds_max_ijk[2]),int(bounds_min_ijk[2])]
    roi_y = [int(bounds_min_ijk[1]),int(bounds_max_ijk[1])]
    roi_z = [int(bounds_min_ijk[0]),int(bounds_max_ijk[0])]
    roi_3D_cord_CT = [np.min(roi_x),np.max(roi_x),np.min(roi_y),np.max(roi_y),np.min(roi_z),np.max(roi_z)]
    roi_3D_cord_CT = np.add(roi_3D_cord_CT, 1)
    #print('roi center-ijk is: '+str(np.round(roiCenter_ijk)))
    #print('roi bound min-ijk is: '+str(np.round(bounds_min_ijk)))
    #print('roi bound max-ijk is: '+str(np.round(bounds_max_ijk)))

    ############################ Seed cort #########################################
    seed_pos = [0.0,0.0,0.0,0.0]
    cortbone_seeds.GetNthFiducialWorldCoordinates(0, seed_pos)
    # change seed coordinates to ijk
    print('seed is: '+str(seed_pos))

    seed_c = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(seed_c)
    seed_pos_ijk = b_min.MultiplyDoublePoint(seed_pos)
    seed_pos_ijk = np.asarray(seed_pos_ijk)

    print('seed-ijk is: '+str(np.round(seed_pos_ijk)))

    seed_pos_ijk = seed_pos_ijk[0:-1]
    seed_pos_ijk_cort = np.round(seed_pos_ijk)

    ############################ Seed trab #########################################
    seed_pos = [0.0,0.0,0.0,0.0]
    trabbone_seeds.GetNthFiducialWorldCoordinates(0, seed_pos)
    # change seed coordinates to ijk
    print('seed is: '+str(seed_pos))

    seed_c = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(seed_c)
    seed_pos_ijk = b_min.MultiplyDoublePoint(seed_pos)
    seed_pos_ijk = np.asarray(seed_pos_ijk)

    print('seed-ijk is: '+str(np.round(seed_pos_ijk)))

    seed_pos_ijk = seed_pos_ijk[0:-1]
    seed_pos_ijk_trab = np.round(seed_pos_ijk)

    ############################ Seed lesion #########################################
    seed_pos = [0.0,0.0,0.0,0.0]
    les_seeds.GetNthFiducialWorldCoordinates(0, seed_pos)
    # change seed coordinates to ijk
    print('seed is: '+str(seed_pos))

    seed_c = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(seed_c)
    seed_pos_ijk = b_min.MultiplyDoublePoint(seed_pos)
    seed_pos_ijk = np.asarray(seed_pos_ijk)

    print('seed-ijk is: '+str(np.round(seed_pos_ijk)))

    seed_pos_ijk = seed_pos_ijk[0:-1]
    seed_pos_ijk_les = np.round(seed_pos_ijk)

    ############################ Seed bone #########################################
    seed_pos = [0.0,0.0,0.0,0.0]
    bone_seeds.GetNthFiducialWorldCoordinates(0, seed_pos)
    # change seed coordinates to ijk
    print('seed is: '+str(seed_pos))

    seed_c = vtk.vtkMatrix4x4()
    inputVolume_ct.GetRASToIJKMatrix(seed_c)
    seed_pos_ijk = b_min.MultiplyDoublePoint(seed_pos)
    seed_pos_ijk = np.asarray(seed_pos_ijk)

    print('seed-ijk is: '+str(np.round(seed_pos_ijk)))

    seed_pos_ijk = seed_pos_ijk[0:-1]
    seed_pos_ijk_bone = np.round(seed_pos_ijk)

    ############################ segmentation algorithm ########################
    # call segmentation
    seed_les_spect = seed_pos_ijk_les
    seed_bone_spect = seed_pos_ijk_bone
    seed_cort_ct = seed_pos_ijk_cort
    seed_trab_ct = seed_pos_ijk_trab

    ################ CT seg
    CT_labeled = self.seg_alg_CT(ct_img, roi_3D_cord_CT, beta)
    
    
    # cort bone
    CT_labeled  = np.ascontiguousarray(CT_labeled, np.float32)
    print('seeds CT:')
    print([int(round(seed_cort_ct[0])), int(round(seed_cort_ct[1])), int(round(seed_cort_ct[2]))])

    print('size CT:')
    print(CT_labeled.shape)
    CT_cort_labeled = self.extract_reg_w_same_label(CT_labeled, int(round(seed_cort_ct[2])), int(round(seed_cort_ct[1])), int(round(seed_cort_ct[0])))

    # trab bone
    CT_trab_labeled = self.extract_reg_w_same_label(CT_labeled, int(round(seed_trab_ct[2])), int(round(seed_trab_ct[1])), int(round(seed_trab_ct[0])))

    # merge regions
    CT_labeled = np.zeros(CT_labeled.shape)
    CT_labeled = CT_labeled + (CT_cort_labeled + CT_trab_labeled)
    CT_labeled[CT_labeled >= 1] = 1

    ################ SPECT seg
    CT_labeled  = np.ascontiguousarray(CT_labeled, np.int16)

    SPECT_labeled = self.seg_alg_SPECT(spect_img, ct_img, CT_labeled, roi_3D_cord_SPECT, roi_3D_cord_CT, beta, gamma)
    

    ################
    SPECT_les_labeled = np.zeros(SPECT_labeled.shape)
    SPECT_labeled_roi = SPECT_labeled[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]]
    zdim,ydim,xdim = SPECT_labeled_roi.shape
    
    SPECT_les_labeled_roi = np.ascontiguousarray(np.asarray(list(SPECT_labeled_roi)), np.int16)
    num_clus_spect = 3
    test_v2Widget.c_func_region_grow_cropped3Dimg(SPECT_les_labeled_roi, xdim, ydim, zdim, int(round(seed_les_spect[0])-roi_3D_cord_CT[4]), int(round(seed_les_spect[1])-roi_3D_cord_CT[2]), int(round(seed_les_spect[2])-roi_3D_cord_CT[0]), num_clus_spect)
    SPECT_les_labeled[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]] = SPECT_les_labeled_roi


    SPECT_bone_labeled = np.zeros(SPECT_labeled.shape)
    SPECT_bone_labeled_roi = self.extract_reg_w_same_label(SPECT_labeled_roi, int(round(seed_bone_spect[2])-roi_3D_cord_CT[0]), int(round(seed_bone_spect[1])-roi_3D_cord_CT[2]), int(round(seed_bone_spect[0])-roi_3D_cord_CT[4]))
    SPECT_bone_labeled[roi_3D_cord_CT[0]:roi_3D_cord_CT[1],roi_3D_cord_CT[2]:roi_3D_cord_CT[3],roi_3D_cord_CT[4]:roi_3D_cord_CT[5]] = SPECT_bone_labeled_roi
    '''
    seedx = seed_les[0]; seedy = seed_les[1]; seedz = seed_les[2]
    print('seed 3d cord is: '+str([seedx, seedy, seedz]))
    img_3D = outputLabel
    imSize = np.shape(img_3D)
    #print('im size: '+str(imSize))
    num_clus = 3
    outputLabel = self.region_grow(img_3D, seedx, seedy, seedz, num_clus, imSize)
    #print('sum is: '+str(np.sum(outputLabel[slice_i,:,:])))
    '''
    slicer.util.updateVolumeFromArray(outputLesClusterVolume,SPECT_les_labeled) # clustering results
    slicer.util.updateVolumeFromArray(outputBoneClusterVolume,SPECT_bone_labeled)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputLesClusterVolume, outputLesSegmentation)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(outputBoneClusterVolume, outputBoneSegmentation) 
    # update volume
    
    

    logging.info('Processing completed')


    return True

class test_v2Test(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_test_v21()

  def test_test_v21(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = test_v2Logic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
