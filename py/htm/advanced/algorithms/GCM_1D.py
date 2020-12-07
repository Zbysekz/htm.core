import numpy
import copy
from htm.bindings.math import Random

class GCM:
    def __init__(self, size):
        self.cells = [0]*size
        self.size = size

    def Shift(self, n):
        if n > 0:
            for i in range(n):
                self.ShiftRight()
        else:
            for i in range(n):
                self.ShiftLeft()

    def ShiftRight(self):
        tmp = self.cells[-1]
        for i in reversed(range(len(self.cells)-1)):
            self.cells[i+1] = self.cells[i]
        self.cells[0] = tmp

    def ShiftLeft(self):
        tmp = self.cells[0]
        for i in reversed(range(len(self.cells)-1)):
            self.cells[i] = self.cells[i + 1]
        self.cells[-1] = tmp

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.cells) != len(other.cells):
                raise RuntimeError("Comparing two GCM's of different lengths!")
            return self.cells == other.cells
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.cells)

class GCM_1D:
    def __init__(self, n=[3,5,8], seed=0):
        self.GCM = [GCM(x) for x in n]
        self.seed = seed

        self.dimensionSize = numpy.lcm.reduce(n)

        self.rng = Random(seed)

    def Shift(self, n):
        for g in self.GCM:
            g.Shift(n)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.GCM) != len(other.GCM):
                raise RuntimeError("Comparing two 1D_GCM's of different lengths!")

            equal = True
            for g in range(len(self.GCM)):
                if self.GCM[g] != other.GCM[g]:
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
        for g in self.GCM:
            txt += str(g)
        return txt
    def activateRandomLocation(self):
        for gcm in self.GCM:
            number = rng.getint()




if __name__ == "__main__":

    GCM1 = GCM_1D()

    GCM1.GCM[0].cells = [1, 0, 0]
    GCM1.GCM[1].cells = [1, 0, 0, 0, 0]
    GCM1.GCM[2].cells = [1, 0, 0, 0, 0, 0, 0, 0]

    GCM2 = GCM_1D()

    GCM2.GCM[0].cells = [1, 0, 0]
    GCM2.GCM[1].cells = [0, 0, 0, 1, 0]
    GCM2.GCM[2].cells = [0, 0, 0, 1, 0, 0, 0, 0]

    print("Dimension size:"+str(GCM1.dimensionSize))
    print("DISTANCE:"+str(GCM1.DistanceFrom(GCM2)))


