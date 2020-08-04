# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2020, Numenta, Inc.
#
# author: David Keeney, dkeeney@gmail.com, July 2020
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

""" 
Unit tests for using NetworkAPI with python. 
Simular to the C++ example napi_hello.cpp
"""

import datetime
import numpy as np
import math
import unittest

from htm.bindings.sdr import SDR
from htm.bindings.engine_internal import Network,Region
from htm.advanced.support.register_regions import registerAllAdvancedRegions

EPOCHS = 3

class NetworkAPI_SineWave_Test(unittest.TestCase):
  """ Unit tests for Network class. """

  def testNetwork(self):
    """
    A simple NetworkAPI example with three regions.
    ///////////////////////////////////////////////////////////////
    //
    //                          .------------------.
    //                         |    encoder        |
    //        sinewave data--->|(RDSEEncoderRegion)|
    //                         |                   |
    //                         `-------------------'
    //                                 |
    //                         .-----------------.
    //                         |     sp          |
    //                         |  (SPRegion)     |
    //                         |                 |
    //                         `-----------------'
    //                                 |
    //                         .-----------------.
    //                         |      tm         |
    //                         |   (TMRegion)    |---->anomaly score
    //                         |                 |
    //                         `-----------------'
    //
    //////////////////////////////////////////////////////////////////
    """
    
    """ Creating network instance. """
    config = """
        {network: [
            {addRegion: {name: "encoder", type: "RDSEEncoderRegion", params: {size: 1000, sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}}},
            {addRegion: {name: "sp", type: "SPRegion", params: {columnCount: 2048, globalInhibition: true}}},
            {addRegion: {name: "tm", type: "TMRegion", params: {cellsPerColumn: 8, orColumnOutputs: true}}},
            {addLink:   {src: "encoder.encoded", dest: "sp.bottomUpIn"}},
            {addLink:   {src: "sp.bottomUpOut", dest: "tm.bottomUpIn"}}
        ]}"""
    net = Network();
    net.configure(config);
    
    # iterate EPOCHS times
    x = 0.00
    for e in range(EPOCHS):
      # Data is a sine wave,    (Note: first iteration is for x=0.01, not 0)
      x += 0.01  # advance one step, 0.01 radians
      s = math.sin(x)   # compute current sine as data.
      #  feed data to RDSE encoder via its "sensedValue" parameter.
      net.getRegion('encoder').setParameterReal64('sensedValue', s)
      net.run(1)  # Execute one iteration of the Network object

    # Retreive the final anomaly score from the TM object's 'anomaly' output. (as a single element numpy array)
    score = np.array(net.getRegion('tm').getOutputArray('anomaly'))
    self.assertEqual(score, [1])
    #Note: Anomaly score will be 1 until there have been enough iterations to 'learn' the pattern.
    
    
  def testGetSpec(self):
    """
    A test of the Network.getSpecJSON( ) function.
    """
    expected = """{"spec": "RDSEEncoderRegion",
  "parameters": {
    "activeBits": {
      "type": "UInt32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0"
    }
    "category": {
      "type": "Bool",
      "count": 1,
      "access": "Create",
      "defaultValue": "false"
    }
    "noise": {
      "description": "amount of noise to add to the output SDR. 0.01 is 1%",
      "type": "Real32",
      "count": 1,
      "access": "ReadWrite",
      "defaultValue": "0.0"
    }
    "radius": {
      "type": "Real32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0.0"
    }
    "resolution": {
      "type": "Real32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0.0"
    }
    "seed": {
      "type": "UInt32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0"
    }
    "sensedValue": {
      "description": "The value to encode. Overriden by input 'values'.",
      "type": "Real64",
      "count": 1,
      "access": "ReadWrite",
      "defaultValue": "0.0"
    }
    "size": {
      "type": "UInt32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0"
    }
    "sparsity": {
      "type": "Real32",
      "count": 1,
      "access": "Create",
      "defaultValue": "0.0"
    }
  },
  "inputs": {
    "values": {
      "description": "Values to encode. Overrides sensedValue.",
      "type": "Real64",
      "count": 1,
      "required": 0,
      "regionLevel": 1,
      "isDefaultInput": 1
    },
  },
  "outputs": {
    "bucket": {
      "description": "Quantized sample based on the radius. Becomes the title for this sample in Classifier.",
      "type": "Real64",
      "count": 1,
      "regionLevel": 0,
      "isDefaultOutput": 0
    },
    "encoded": {
      "description": "Encoded bits. Not a true Sparse Data Representation (SP does that).",
      "type": "SDR",
      "count": 0,
      "regionLevel": 1,
      "isDefaultOutput": 1
    },
  }
}"""
    net = Network()
    json = net.getSpecJSON("RDSEEncoderRegion");
    self.assertEqual(json, expected)
    
    
  def testGetParameters(self):
    """
    A test of the getParameters( ) and getParameterJSON() functions.
    """
    expected = """{
  "activeBits": 200,
  "category": false,
  "noise": 0.010000,
  "radius": 0.030000,
  "resolution": 0.000150,
  "seed": 2019,
  "sensedValue": 0.000000,
  "size": 1000,
  "sparsity": 0.200000,
}"""

    pars = dict(
      moduleCount=dict(
        description="Number of grid cell modules",
        dataType="UInt32",
        accessMode="Create",
        count=1
      ),
      cellsPerAxis=dict(
        description="Determines the number of cells. Determines how space is "
                    "divided between the cells",
        dataType="UInt32",
        accessMode="Create",
        count=1
      ),
      scale=dict(
        description="Determines the amount of world space covered by all of "
                    "the cells combined. In grid cell terminology, this is "
                    "equivalent to the 'scale' of a module. One scale value "
                    "for each grid cell module. Array size must match "
                    "'moduleCount' parameter",
        dataType="Real32",
        accessMode="Create",
        count=0,
      ),
      orientation=dict(
        description="The rotation of this map, measured in radians. One "
                    "orientation value for each grid cell module. Array size "
                    "must match 'moduleCount' parameter",
        dataType="Real32",
        accessMode="Create",
        count=0,
      ),
      anchorInputSize=dict(
        description="The number of input bits in the anchor input",
        dataType="UInt32",
        accessMode="Create",
        count=1,
      ),
      activeFiringRate=dict(
        description="Between 0.0 and 1.0. A cell is considered active if its "
                    "firing rate is at least this value",
        dataType="Real32",
        accessMode="Create",
        count=1,
      ),
      bumpSigma=dict(
        description="Specifies the diameter of a gaussian bump, in units of "
                    "'rhombus edges'. A single edge of the rhombus has length "
                    "1, and this bumpSigma would typically be less than 1. We "
                    "often use 0.18172 as an estimate for the sigma of a rat "
                    "entorhinal bump",
        dataType="Real32",
        accessMode="Create",
        count=1,
        defaultValue="0.18172",
      ),
      activationThreshold=dict(
        description="If the number of active connected synapses on a "
                    "segment is at least this threshold, the segment "
                    "is said to be active",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        constraints="",
        defaultValue="10"
      ),
      initialPermanence=dict(
        description="Initial permanence of a new synapse",
        accessMode="Create",
        dataType="Real32",
        count=1,
        constraints="",
        defaultValue="0.21"
      ),
      connectedPermanence=dict(
        description="If the permanence value for a synapse is greater "
                    "than this value, it is said to be connected",
        accessMode="Create",
        dataType="Real32",
        count=1,
        constraints="",
        defaultValue="0.50"
      ),
      learningThreshold=dict(
        description="Minimum overlap required for a segment to learned",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        defaultValue="10"
      ),
      sampleSize=dict(
        description="The desired number of active synapses for an "
                    "active cell",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        defaultValue="20"
      ),
      permanenceIncrement=dict(
        description="Amount by which permanences of synapses are "
                    "incremented during learning",
        accessMode="Create",
        dataType="Real32",
        count=1,
        defaultValue="0.1"
      ),
      permanenceDecrement=dict(
        description="Amount by which permanences of synapses are "
                    "decremented during learning",
        accessMode="Create",
        dataType="Real32",
        count=1,
        defaultValue="0.0"
      ),
      maxSynapsesPerSegment=dict(
        description="The maximum number of synapses per segment",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        defaultValue="-1"
      ),
      learningMode=dict(
        description="A boolean flag that indicates whether or not we should "
                    "learn by associating the location with the sensory "
                    "input",
        dataType="Bool",
        accessMode="ReadWrite",
        count=1,
        defaultValue="False"
      ),
      dualPhase=dict(
        description="A boolean flag that indicates whether or not we should "
                    "process movement and sensation using two phases of the "
                    "same network. When this flag is enabled, the compute "
                    "method will alternate between movement and sensation on "
                    "each phase. When this flag is disabled both movement and "
                    "sensations will be processed on a single phase.",
        dataType="Bool",
        accessMode="ReadWrite",
        count=1,
        defaultValue="True"
      ),
      dimensions=dict(
        description="The number of dimensions represented in the displacement",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        defaultValue="2"
      ),
      seed=dict(
        description="Seed for the random number generator",
        accessMode="Create",
        dataType="UInt32",
        count=1,
        defaultValue="42"
      )
    )
    import json
    registerAllAdvancedRegions()
    net = Network()
    encoder = net.addRegion("asd", "py.GridCellLocationRegion",#"py.ColumnPoolerRegion"
                            """
                            {"moduleCount": 10,
                            "dimensions": 3,
                            "activationThreshold": 8,
                            "initialPermanence": 1.0,
                            "connectedPermanence": 0.5,
                            "learningThreshold": 8,
                            "sampleSize": 10,
                            "permanenceIncrement": 0.1,
                            "permanenceDecrement": 0.0,
                            "cellsPerAxis":6,
                            
                            "anchorInputSize":20,
                            "activeFiringRate":0.0
                            }
                            """

                            # "scale":[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            #         "orientation": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

                            #                             """{
# 	"activationThresholdDistal": 20,
# 	"cellCount": 4096,
# 	"connectedPermanenceDistal": 0.5,
# 	"connectedPermanenceProximal": 0.5,
# 	"initialDistalPermanence": 0.51,
# 	"initialProximalPermanence": 0.6,
# 	"minThresholdProximal": 5,
# 	"sampleSizeDistal": 30,
# 	"sampleSizeProximal": 10,
# 	"sdrSize": 40,
# 	"synPermDistalDec": 0.001,
# 	"synPermDistalInc": 0.1,
# 	"synPermProximalDec": 0.001,
# 	"synPermProximalInc": 0.1
# }"""
)

    #encoder = net.addRegion("encoder", "RDSEEncoderRegion", "{size: 1000, sparsity: 0.2, radius: 0.03, seed: 2019, noise: 0.01}")
    json = encoder.getParameters()
    print(json)
    return
    self.assertEqual(json, expected)
    json = encoder.getParameterJSON("activeBits", False)
    self.assertEqual(json, "200")
    json = encoder.getParameterJSON("activeBits", True)
    self.assertEqual(json, "{\"value\": 200, \"type\": \"UInt32\"}")
    


if __name__ == "__main__":
  unittest.main()
