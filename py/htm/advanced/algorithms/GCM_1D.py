import copy
import math
import numpy as np
from htm.advanced.support import numpy_helpers as np2
from htm.advanced.algorithms.connections import Connections
from htm.bindings.math import Random
from htm.bindings.sdr import SDR


class SUBGCM:
    def __init__(self, size):
        self.cells = [0]*size
        self.size = size

    def Shift(self, n):
        if n > 0:
            for i in range(n):
                self.ShiftRight()
        else:
            for i in range(abs(n)):
                self.ShiftLeft()

    def ShiftRight(self):
        tmp = self.cells[-1]
        for i in reversed(range(len(self.cells)-1)):
            self.cells[i+1] = self.cells[i]
        self.cells[0] = tmp

    def ShiftLeft(self):
        tmp = self.cells[0]
        for i in range(len(self.cells)-1):
            self.cells[i] = self.cells[i + 1]
        self.cells[-1] = tmp

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.cells) != len(other.cells):
                raise RuntimeError("Comparing two SUBGCM's of different lengths!")
            return self.cells == other.cells
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.cells)

class GCM_1D:
    def __init__(self,
                 anchorInputSize=0,
                 activationThreshold=10,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 learningThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 maxSegmentsPerCell=255,
                 seed=42, n=[3,5,8], scale = 1.0
                 ):
        self.SUBGCMS = [SUBGCM(x) for x in n]
        self.cellCount = sum(n)
        self.activeCells = np.empty(0, dtype="int")
        self.sensoryAssociatedCells = np.empty(0, dtype="int")

        self.anchorInputSize = anchorInputSize

        self.displacementRemainder = 0.0
        self.scale = scale
        self.seed = seed

        self.dimensionSize = np.lcm.reduce(n) # minimal common multiple

        self.connections = Connections(self.cellCount, connectedPermanence, False)

        # The cells that were activated by sensory input in an inference timestep,
        # or cells that were associated with sensory input in a learning timestep.
        self.sensoryAssociatedCells = np.empty(0, dtype="int")

        self.activeSegments = np.empty(0, dtype="uint32")

        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.learningThreshold = learningThreshold
        self.sampleSize = sampleSize
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.activationThreshold = activationThreshold
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell

        self.rng = Random(seed)

    def Shift(self, n):
        for g in self.SUBGCMS:
            g.Shift(n)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.SUBGCMS) != len(other.SUBGCMS):
                raise RuntimeError("Comparing two 1D_GCM's of different lengths!")

            equal = True
            for g in range(len(self.SUBGCMS)):
                if self.SUBGCMS[g] != other.SUBGCMS[g]:
                    equal = False
                    break
            return equal
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def DistanceFrom(self, otherGCM):
        myCopy = copy.deepcopy(self)

        maxSize = myCopy.dimensionSize
        ptr = 0
        while myCopy!=otherGCM and ptr < maxSize:
            myCopy.Shift(1)
            ptr += 1
            print(myCopy)
        if ptr == maxSize:
            raise RuntimeError("Not possible to calculate distance!")
        return ptr

    def __str__(self):
        txt = "----GCM-----\n"
        for g in self.SUBGCMS:
            txt += str(g)
        return txt

    def activateRandomLocation(self):
        """
            Set the location to a random point.
        """
        for gcm in self.SUBGCMS:
            r = self.rng.getUInt32(gcm.size)
            gcm.cells = [0] * gcm.size
            gcm.cells[r] = 1

    def reset(self):
        """
        Clear the active cells.
        """
        self.activeCells = np.empty(0, dtype="int")
        self.sensoryAssociatedCells = np.empty(0, dtype="int")


    def movementCompute(self, displacement, noiseFactor=0):
        """
        Shift the current active cells by a vector.
        This is called when the sensor moves.

        @param 1D displacement (float)
        A translation vector.
        """
        self.displacementRemainder += displacement * self.scale
        self.Shift(math.floor(self.displacementRemainder))
        self.displacementRemainder -= math.floor(self.displacementRemainder)


    def _sensoryComputeInferenceMode(self, anchorInput):
        """
        Infer the location from sensory input. Activate any cells with enough active
        synapses to this sensory input. Deactivate all other cells.

        @param anchorInput (numpy array)
        A sensory input. This will often come from a feature-location pair layer.
        """
        if len(anchorInput.sparse) == 0:
            return

    def _sensoryComputeLearningMode(self, anchorInput):
        """
        Associate this location with a sensory input. Subsequently, anchorInput will
        activate the current location during anchor().

        @param anchorInput SDR
        A sensory input. This will often come from a feature-location pair layer.
        """
        overlaps, potentialOverlaps = self.connections.computeActivityFull(anchorInput, False)
        activeSegments = np.flatnonzero(overlaps >= self.activationThreshold)
        matchingSegments = np.flatnonzero(potentialOverlaps >= self.learningThreshold)

        # Cells with a active segment: reinforce the segment
        cellsForActiveSegments = self.connections.mapSegmentsToCells(activeSegments)
        learningActiveSegments = activeSegments[
            np.in1d(cellsForActiveSegments, self.getLearnableCells(), assume_unique=True)]
        remainingCells = np.setdiff1d(self.getLearnableCells(), cellsForActiveSegments, assume_unique=True)

        # Remaining cells with a matching segment: reinforce the best
        # matching segment.
        candidateSegments = self.connections.filterSegmentsByCell(matchingSegments, remainingCells)
        cellsForCandidateSegments = (self.connections.mapSegmentsToCells(candidateSegments))
        candidateSegments = candidateSegments[np.in1d(cellsForCandidateSegments, remainingCells, assume_unique=True)]
        onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments], cellsForCandidateSegments)
        learningMatchingSegments = candidateSegments[onePerCellFilter]

        newSegmentCells = np.setdiff1d(remainingCells, cellsForCandidateSegments, assume_unique=True)

        for learningSegments in (learningActiveSegments, learningMatchingSegments):
            self._learn(learningSegments, anchorInput, potentialOverlaps)

        # Remaining cells without a matching segment: grow one.
        self._learnOnNewSegments(newSegmentCells, anchorInput)

        self.activeSegments = activeSegments
        self.sensoryAssociatedCells = self.getLearnableCells()

    def sensoryCompute(self, anchorInput, anchorGrowthCandidates, learn):
        """
        This is called when the sensor senses something
        """
        anchorInputSDR = SDR(self.anchorInputSize)

        if learn:
            anchorInputSDR.sparse = anchorGrowthCandidates
            self._sensoryComputeLearningMode(anchorInputSDR)

        else:
            anchorInputSDR.sparse = anchorInput
            self._sensoryComputeInferenceMode(anchorInputSDR)

    def _learn(self, learningSegments, activeInput, potentialOverlaps):
        """
        Adjust synapse permanences, grow new synapses, and grow new segments.

        @param learningActiveSegments (numpy array)
        @param learningMatchingSegments (numpy array)
        @param segmentsToPunish (numpy array)
        @param activeInput SDR
        @param potentialOverlaps (numpy array)
        """
        for segment in learningSegments:
            # Learn on existing segments
            self.connections.adaptSegment(segment, activeInput, self.permanenceIncrement, self.permanenceDecrement,
                                          False)

            # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
            # grow per segment. "maxNew" might be a number or it might be a list of
            # numbers.
            if self.sampleSize == -1:
                maxNew = len(activeInput.sparse)
            else:
                maxNew = self.sampleSize - potentialOverlaps[segment]

            if self.maxSynapsesPerSegment != -1:
                synapseCounts = self.connections.numSynapses(segment)
                numSynapsesToReachMax = self.maxSynapsesPerSegment - synapseCounts
                maxNew = np.where(maxNew <= numSynapsesToReachMax, maxNew, numSynapsesToReachMax)

            if maxNew > 0:
                self.connections.growSynapses(segment, activeInput.sparse, self.initialPermanence, self.rng, maxNew)

    def _learnOnNewSegments(self, newSegmentCells, growthCandidates):

        numNewSynapses = len(growthCandidates.sparse)

        if self.sampleSize != -1:
            numNewSynapses = min(numNewSynapses, self.sampleSize)

        if self.maxSynapsesPerSegment != -1:
            numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)

        for cell in newSegmentCells:
            newSegment = self.connections.createSegment(cell, self.maxSegmentsPerCell)
            self.connections.growSynapses(newSegment, growthCandidates.sparse, self.initialPermanence, self.rng,
                                          numNewSynapses)

    def getActiveCells(self):
        # for now, dummy like this
        simpleList = []
        for x in self.SUBGCMS:
            simpleList+= x.cells

        activeIndexes = [i for i, value in enumerate(simpleList) if value]
        self.activeCells = np.empty(0, dtype="int")
        self.activeCells = np.append(self.activeCells, activeIndexes)

        return self.activeCells

    def getLearnableCells(self):
        # note in Gaussian 2D module, it was np.where(cellExcitations == cellExcitations.max())[0]
        return self.getActiveCells()

    def getSensoryAssociatedCells(self):
        return self.sensoryAssociatedCells

    def numberOfCells(self):
        return self.cellCount


if __name__ == "__main__":

    GCM1 = GCM_1D()

    GCM1.SUBGCMS[0].cells = [1, 0, 0]
    GCM1.SUBGCMS[1].cells = [1, 0, 0, 0, 0]
    GCM1.SUBGCMS[2].cells = [1, 0, 0, 0, 0, 0, 0, 0]

    GCM2 = GCM_1D()

    GCM2.SUBGCMS[0].cells = [1, 0, 0]
    GCM2.SUBGCMS[1].cells = [0, 0, 0, 1, 0]
    GCM2.SUBGCMS[2].cells = [0, 0, 0, 1, 0, 0, 0, 0]

    print("Dimension size:"+str(GCM1.dimensionSize))
    print("DISTANCE:"+str(GCM1.DistanceFrom(GCM2)))

    print(GCM1)
    GCM1.activateRandomLocation()

    print(GCM1)


