
#############################################################################################
# 1D Morphodynamic Model for the Yellow River
#            using a coupled approach
#                         Developed by Junqiang Xia
#     						   May-July 2010
#  Characteristics of this model:
#  (1) Quasi-2D or 1D,Different Roughness between Main Channel and Flood Plain
#  (2) Nonequlibrium transport of nonuniform Suspended Sediments
#  (3) Adapt to the Low and High Sediment Concentrations
#  (4) unsteady Flow and coupled solution
#############################################################################################

import time
import numpy as np


class MorphodynamicModel1D:
    def __init__(self):
        self.Nshow = False
        self.MaxNcs = 100  # Maximum number of cross-sections
        self.MaxNGS = 10   # Maximum number of sediment groups
        self.Maxstp = 1000 # Maximum number of time steps
        self.Maxcts = 100  # Maximum number of control cross-sections
        self.MaxTby = 10   # Maximum number of tributaries
        self.MaxQN = 10    # Maximum number of flow rates for roughness calculation

        # Import variable arrays
        self.import_var_arr()

    def import_var_arr(self):
        # Define arrays
        self.NOCSSP = np.zeros(self.MaxNcs, dtype=int)  # Cross-sections with bed gradation
        self.PBEDSP = np.zeros((self.MaxNcs, self.MaxNGS))  # Bed gradation at cross-sections
        self.DIstSP = np.zeros(self.MaxNcs)  # Distance of cross-sections with bed gradation
        self.DLtemp = np.zeros(self.MaxNcs)  # Temporary array for distance
        self.PKtemp = np.zeros(self.MaxNcs)  # Temporary array for gradation
        self.TimeWDrch = np.zeros((self.Maxstp, self.Maxcts))  # Flow rate at control cross-sections
        self.TimeSDrch = np.zeros((self.Maxstp, self.Maxcts))  # Sediment load at control cross-sections
        self.TimeTT = np.zeros(self.Maxstp)  # Time array
        self.TimePP = np.zeros(self.Maxstp)  # Time array
        self.TimeSPK = np.zeros((self.Maxstp, self.MaxNGS))  # Suspended sediment gradation in main channel
        self.TimePKtby = np.zeros((self.MaxTby, self.Maxstp, self.MaxNGS))  # Suspended sediment gradation in tributaries
        self.TSPMC = np.zeros(self.Maxstp)  # Time array for suspended sediment in main channel
        self.TSPKMC = np.zeros((self.Maxstp, self.MaxNGS))  # Suspended sediment gradation in main channel
        self.NSPtby = np.zeros(self.MaxTby, dtype=int)  # Number of tributaries
        self.TSPtby = np.zeros((self.MaxTBY, self.Maxstp))  # Time array for suspended sediment in tributaries
        self.TSPKtby = np.zeros((self.MaxTBY, self.Maxstp, self.MaxNGS))  # Suspended sediment gradation in tributaries
        self.RNCTS = np.zeros((self.MaxCTS, self.MaxQN))  # Roughness of main channel at control cross-sections
        self.RNLWFP = np.zeros(self.MaxCTS)  # Roughness of low floodplain at control cross-sections
        self.RNhgFP = np.zeros(self.MaxCTS)  # Roughness of high floodplain at control cross-sections
        self.DHbed = np.zeros(self.MaxCTS)  # Roughness increment at control cross-sections
        self.DNbed = np.zeros(self.MaxCTS)  # Bed deformation thickness at control cross-sections
        self.TempXX = np.zeros(self.MaxNcs)  # Temporary array for coordinates
        self.TempYY = np.zeros(self.MaxNcs)  # Temporary array for coordinates
        self.BWMC = np.zeros(self.MaxNcs)  # Width of main channel
        self.ZBBF = np.zeros(self.MaxNcs)  # Elevation of floodplain

    def input_data(self):
        # Import modules
        import os

        # Define local variables
        NOCSSP = np.zeros(self.MaxNcs, dtype=int)
        PBEDSP = np.zeros((self.MaxNcs, self.MaxNGS))
        DIstSP = np.zeros(self.MaxNcs)
        DLtemp = np.zeros(self.MaxNcs)
        PKtemp = np.zeros(self.MaxNcs)
        TimeWDrch = np.zeros((self.Maxstp, self.Maxcts))
        TimeSDrch = np.zeros((self.Maxstp, self.Maxcts))
        TimeTT = np.zeros(self.Maxstp)
        TimePP = np.zeros(self.Maxstp)
        TimeSPK = np.zeros((self.Maxstp, self.MaxNGS))
        TimePKtby = np.zeros((self.MaxTby, self.Maxstp, self.MaxNGS))
        TSPMC = np.zeros(self.Maxstp)
        TSPKMC = np.zeros((self.Maxstp, self.MaxNGS))
        NSPtby = np.zeros(self.MaxTby, dtype=int)
        TSPtby = np.zeros((self.MaxTBY, self.Maxstp))
        TSPKtby = np.zeros((self.MaxTBY, self.Maxstp, self.MaxNGS))
        RNCTS = np.zeros((self.MaxCTS, self.MaxQN))
        RNLWFP = np.zeros(self.MaxCTS)
        RNhgFP = np.zeros(self.MaxCTS)
        DHbed = np.zeros(self.MaxCTS)
        DNbed = np.zeros(self.MaxCTS)
        TempXX = np.zeros(self.MaxNcs)
        TempYY = np.zeros(self.MaxNcs)
        BWMC = np.zeros(self.MaxNcs)
        ZBBF = np.zeros(self.MaxNcs)

        print('Determine the output dir')
        with open('Filename.Dat', 'r') as f:
            f.readline()  # Skip blank line
            Filnam = f.readline().strip()
            f.readline()  # Skip blank line
            Directory = f.readline().strip('"')
            Dirlnt = len(Directory)

        print('1---File=Filename.Dat has been input')

        # Input Namlen
        Namlen = self.fileinp(Filnam)

        # Input basic model parameters
        Fullnam = Filnam[:Namlen] + '_ALPRMT.Dat'
        with open(Fullnam, 'r') as f:
            f.readline()  # Skip blank line

            # Read common parameters
            f.readline()  # Skip header
            Imax, Jmax = [int(x) for x in f.readline().split()]
            TimeSM = float(f.readline())
            NYSPin, TimeSP = [int(x) for x in f.readline().split()]
            DTstep = float(f.readline())
            NTRD = int(f.readline())
            BankCal = int(f.readline())
            NumTBY = int(f.readline())
            NCTCS = int(f.readline())
            NumGS = int(f.readline())
            NCPFS = int(f.readline())
            MDRGH = int(f.readline())
            NStart = int(f.readline())
            OutPutGW = int(f.readline())
            print('11111111111111111')

            # Read parameters for flow calculation
            f.readline()  # Skip header
            GRAV = float(f.readline())
            CitaFw = float(f.readline())
            Hmin = float(f.readline())
            DHmin = float(f.readline())
            Qmin = float(f.readline())
            ITSUMF = int(f.readline())
            EPSQD = float(f.readline())
            EPSZW = float(f.readline())
            print('222222222222222222')

            # Read parameters for sediment transport calculation
            f.readline()  # Skip header
            NFLST = int(f.readline())
            NFLBD = int(f.readline())
            NFLGA = int(f.readline())
            PFlow = float(f.readline())
            Psedi = float(f.readline())
            Pdry = float(f.readline())
            ZWMVL = float(f.readline())
            ITsumS = int(f.readline())
            ERRSD = float(f.readline())
            CitaSD = float(f.readline())
            CokZ, ComZ = [float(x) for x in f.readline().split()]
            CokW, ComW = [float(x) for x in f.readline().split()]
            CokZJ, ComZJ = [float(x) for x in f.readline().split()]
            CokV, ComV = [float(x) for x in f.readline().split()]
            MDSCC = int(f.readline())
            CitaDPS = float(f.readline())
            CoFaDEG, COFbDEG = [float(x) for x in f.readline().split()]
            COFaDEP, COFbDEP = [float(x) for x in f.readline().split()]
            MDLDA = int(f.readline())
            MaxNML, MinNML, NMLini = [int(x) for x in f.readline().split()]
            Dmix, Dmem = [float(x) for x in f.readline().split()]
            DZmor = float(f.readline())
            THmin = float(f.readline())
            print('333333333333333333333333')

        print('2---File=*_AllPara.Dat has been input')

        Nstep = int(TimeSM * 3600.0 / DTstep)  # Total number of time steps

        # Input cross-section geometry and distance
        Fullnam = Filnam[:Namlen] + '_CSProf.Dat'
        print(Fullnam)
        with open(Fullnam, 'r') as f:
            f.readline()  # Skip blank line
            Imax0 = int(f.readline())  # Total number of cross-sections

            print(f'Imax, Imax0 = {Imax}, {Imax0}')  # Check data

            for i in range(1, Imax + 1):
                f.readline()  # Skip header
                f.readline()  # Skip specification
                NPT1CS = [int(x) for x in f.readline().split()]  # Number of nodes in cross-section
                f.readline()  # Skip floodplain elevations

                for j in range(NPT1CS[i - 1]):
                    j1, XXIJ[i, j], ZBIJ[i, j], KNIJ[i, j] = [float(x) for x in f.readline().split()]
                    ZBINL[i, j] = ZBIJ[i, j]
                    XXIJNL[i, j] = XXIJ[i, j]

            f.readline()  # Skip separator
            f.readline()  # Skip specification
            for i in range(1, Imax + 1):
                IX, DISTLG[i - 1] = [float(x) for x in f.readline().split()]  # Distance along the river

            f.readline()  # Skip separator
            NCTCS = int(f.readline())  # Number of control cross-sections
            f.readline()  # Skip specification
            for i in range(NCTCS):
                Ix, IPCTCS[i] = [int(x) for x in f.readline().split()]
                print(Ix, IPCTCS[i])

        print('3---File=*_CSProf.Dat has been input')

        # Determine the sub-domains of floodplain and main channel
        self.DomainsFPMC()

    def DomainsFPMC(self):
        # Determine the sub-domains of floodplain and main channel
        # NMC1CS(i)   =  Number of main channels in each cross-section
        # NodMCL(i,k) =  Left node of main channel
        # NodMCR(i,k) =  Right node of main channel
        for i in range(1, self.Imax + 1):
            print(i, self.NMC1CS[i - 1], [(self.NodMCL[i - 1, k - 1], self.NodMCR[i - 1, k - 1]) for k in range(1, self.NMC1CS[i - 1] + 1)])

    def fileinp(self, Filnam):
        # Read the length of the file name
        with open(Filnam, 'r') as f:
            Namlen = len(f.readline().strip())
        return Namlen

if __name__ == "__main__":
    model = MorphodynamicModel1D()
    model.input_data()

