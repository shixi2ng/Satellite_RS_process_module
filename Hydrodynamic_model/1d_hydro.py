import numpy as np

# Module Variables and Arrays in a 1D unSteady Morphological Model for LYR
# Setup common Variables and Arrays in this program
# Finished in May-July, 2010 by XiaJQ
# Contact information:

class Varandarray:

    def __init__(self):
        self.Filename = [''] * 100
        self.Fullnam = ''  # 输出文件路径名 ForExample: fullnam=Directory(:Dirlnt)//'_Zmax.LPT'
        self.Filnam = ''  # 输入文件路径名 ForExample: filnam ='temp\YLR'   !此处文件名可修改 99999
        self.Directory = ''  # 输入文件的目录
        self.Namlen = 0
        self.Dirlnt = 0


        self.Coks = 0.0
        self.Coms = 0.0

        self.NPT1CS = [0] * self.MaxNCS
        self.NMC1CS = [0] * self.MaxNCS
        self.NodMCL = [[0] * self.MaxNMC for _ in range(self.MaxNCS)]
        self.NodMCR = [[0] * self.MaxNMC for _ in range(self.MaxNCS)]
        self.IPCTCS = [0] * self.MaxCTS
        self.NCSTBY = [0] * self.MaxTBY
        self.NTPML = [0] * self.MaxNCS
        self.NumQtby = [0] * self.MaxStp
        self.NumStby = [0] * self.MaxStp
        self.DISTLG = [0.0] * self.MaxNCS
        self.Dx2cs = [0.0] * self.MaxNCS
        self.DistCTS = [0.0] * self.MaxCTS
        self.DM1FRT = [0.0] * self.MaxNGS
        self.DMSiev = [0.0] * self.MaxNGS
        self.TimeTQ = [0.0] * self.MaxStp
        self.TimeQint = [0.0] * self.MaxStp
        self.TimeTS = [0.0] * self.MaxStp
        self.TimeSint = [0.0] * self.MaxStp
        self.TimeTout = [0.0] * self.MaxStp
        self.TimeZout = [0.0] * self.MaxStp
        self.TimeQout = [0.0] * self.MaxStp
        self.TimeSkint = [[0.0] * self.MaxNGS for _ in range(self.MaxStp)]
        self.TimeTMP = [0.0] * self.MaxStp
        self.TimeWtemp = [0.0] * self.MaxStp
        self.TimeTdiv = [0.0] * self.MaxStp
        self.TimeTSdiv = [0.0] * self.MaxStp
        self.TimeWdiv = [[0.0] * self.MaxNCS for _ in range(self.MaxStp)]
        self.TimeSdiv = [[0.0] * self.MaxNCS for _ in range(self.MaxStp)]
        self.QLold = [0.0] * self.MaxNCS
        self.QLnew = [0.0] * self.MaxNCS
        self.TimeTStby = [[0.0] * self.MaxStp for _ in range(self.MaxTBY)]
        self.TimeStby = [[0.0] * self.MaxStp for _ in range(self.MaxTBY)]
        self.TimeSKtby = [[[0.0] * self.MaxNGS for _ in range(self.MaxStp)] for _ in range(self.MaxTBY)]
        self.TimeQSKtby = [[[0.0] * self.MaxNGS for _ in range(self.MaxStp)] for _ in range(self.MaxTBY)]
        self.TimeTQtby = [[0.0] * self.MaxStp for _ in range(self.MaxTBY)]
        self.TimeQtby = [[0.0] * self.MaxStp for _ in range(self.MaxTBY)]
        self.CSZW = [0.0] * self.MaxNCS
        self.CSQQ = [0.0] * self.MaxNCS
        self.CSUU = [0.0] * self.MaxNCS
        self.CSHH = [0.0] * self.MaxNCS
        self.CSAA = [0.0] * self.MaxNCS
        self.CSBB = [0.0] * self.MaxNCS
        self.CSZB = [0.0] * self.MaxNCS
        self.CSSP = [0.0] * self.MaxNCS
        self.CSRN = [0.0] * self.MaxNCS
        self.CSUT = [0.0] * self.MaxNCS
        self.CSWP = [0.0] * self.MaxNCS
        self.CSRD = [0.0] * self.MaxNCS
        self.CSFR = [0.0] * self.MaxNCS
        self.CSQK = [0.0] * self.MaxNCS
        self.CSQM = [0.0] * self.MaxNCS
        self.CSUM = [0.0] * self.MaxNCS
        self.CSHM = [0.0] * self.MaxNCS
        self.CSAM = [0.0] * self.MaxNCS
        self.CSBM = [0.0] * self.MaxNCS
        self.CSQf = [0.0] * self.MaxNCS
        self.CSUf = [0.0] * self.MaxNCS
        self.CSHf = [0.0] * self.MaxNCS
        self.CSAf = [0.0] * self.MaxNCS
        self.CSBf = [0.0] * self.MaxNCS
        self.CSZBmn = [0.0] * self.MaxNCS
        self.CSZBav = [0.0] * self.MaxNCS
        self.CSZBav0 = [0.0] * self.MaxNCS
        self.CSSUS = [0.0] * self.MaxNCS
        self.CSSCC = [0.0] * self.MaxNCS
        self.SUSD50 = [0.0] * self.MaxNCS
        self.SUSDPJ = [0.0] * self.MaxNCS
        self.CSSVL = [0.0] * self.MaxNCS
        self.CSSVM = [0.0] * self.MaxNCS
        self.CSCNK = [0.0] * self.MaxNCS
        self.WSmed = [0.0] * self.MaxNCS
        self.DBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DKIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.AAIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.QKIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.QQIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.UUij = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.BBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.HHij = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.UTIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.WPIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.RDIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.XXIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.ZBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.CSZW = [0.0] * self.MaxNCS
        self.CSQQ = [0.0] * self.MaxNCS
        self.CSUU = [0.0] * self.MaxNCS
        self.CSHH = [0.0] * self.MaxNCS
        self.CSAA = [0.0] * self.MaxNCS
        self.CSBB = [0.0] * self.MaxNCS
        self.CSZB = [0.0] * self.MaxNCS
        self.CSSP = [0.0] * self.MaxNCS
        self.CSRN = [0.0] * self.MaxNCS
        self.CSUT = [0.0] * self.MaxNCS
        self.CSWP = [0.0] * self.MaxNCS
        self.CSRD = [0.0] * self.MaxNCS
        self.CSFR = [0.0] * self.MaxNCS
        self.CSQK = [0.0] * self.MaxNCS
        self.CSQM = [0.0] * self.MaxNCS
        self.CSUM = [0.0] * self.MaxNCS
        self.CSHM = [0.0] * self.MaxNCS
        self.CSAM = [0.0] * self.MaxNCS
        self.CSBM = [0.0] * self.MaxNCS
        self.CSQf = [0.0] * self.MaxNCS
        self.CSUf = [0.0] * self.MaxNCS
        self.CSHf = [0.0] * self.MaxNCS
        self.CSAf = [0.0] * self.MaxNCS
        self.CSBf = [0.0] * self.MaxNCS
        self.CSZBmn = [0.0] * self.MaxNCS
        self.CSZBav = [0.0] * self.MaxNCS
        self.CSZBav0 = [0.0] * self.MaxNCS
        self.CSSUS = [0.0] * self.MaxNCS
        self.CSSCC = [0.0] * self.MaxNCS
        self.SUSD50 = [0.0] * self.MaxNCS
        self.SUSDPJ = [0.0] * self.MaxNCS
        self.CSSVL = [0.0] * self.MaxNCS
        self.CSSVM = [0.0] * self.MaxNCS
        self.CSCNK = [0.0] * self.MaxNCS
        self.WSmed = [0.0] * self.MaxNCS
        self.DBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DKIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.AAIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.QKIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.QQIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.UUij = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.BBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.HHij = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.UTIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.WPIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.RDIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.XXIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.ZBIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DFCIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.TCnode = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.COKSD = [0.0] * self.MaxNCS
        self.COMSD = [0.0] * self.MaxNCS
        self.WSET = [0.0] * self.MaxNCS
        self.WSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.UCIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.SKint = [0.0] * self.MaxStp
        self.ALFSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.BLTSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.SUSIK1 = [0.0] * self.MaxNCS
        self.SUSIK2 = [0.0] * self.MaxNCS
        self.SUSIK3 = [0.0] * self.MaxNCS
        self.SUSIK4 = [0.0] * self.MaxNCS
        self.SUSIK5 = [0.0] * self.MaxNCS
        self.SUSIK6 = [0.0] * self.MaxNCS
        self.SUSIK7 = [0.0] * self.MaxNCS
        self.SUSIK8 = [0.0] * self.MaxNCS
        self.SUSIK9 = [0.0] * self.MaxNCS
        self.SUSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.SCCIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.DPSCCIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.DPSUSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.QSLold = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.QSLnew = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.CSASIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.CSBSIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.BED50 = [0.0] * self.MaxNCS
        self.BED90 = [0.0] * self.MaxNCS
        self.BedPJ = [0.0] * self.MaxNCS
        self.BED10 = [0.0] * self.MaxNCS
        self.BED60 = [0.0] * self.MaxNCS
        self.CSSL1 = [0.0] * self.MaxNCS
        self.CSSL2 = [[0.0] * self.MaxTBY for _ in range(self.MaxNCS)]
        self.CSDA = [0.0] * self.MaxNCS
        self.DAMC = [0.0] * self.MaxNCS
        self.DAIJ = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DZIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.DAIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.DHnode = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DZnode = [[0.0] * self.MaxNPT for _ in range(self.MaxNCS)]
        self.DZMC = [0.0] * self.MaxNCS
        self.dASdt1 = [0.0] * self.MaxNCS
        self.dASdt2 = [0.0] * self.MaxNCS
        self.PDASIK1 = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.PDASIK2 = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.DASIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.VOLCSold = [0.0] * (self.MaxNCS - 1)
        self.VOLMCold = [0.0] * (self.MaxNCS - 1)
        self.TimeVLcs = [0.0] * self.MaxNCS
        self.TimeVLMC = [0.0] * self.MaxNCS
        self.TimeVL2CTCS = [[0.0] * self.MaxCTS for _ in range(self.MaxNCS)]
        self.TimeVL2CTMC = [[0.0] * self.MaxCTS for _ in range(self.MaxNCS)]
        self.CofNDX = [0.0] * self.MaxNCS
        self.CofNDY = [0.0] * self.MaxNCS
        self.DPBEDIK = [[0.0] * self.MaxNGS for _ in range(self.MaxNCS)]
        self.THBAL = [0.0] * self.MaxNCS
        self.THEML = [[0.0] * MaxBML for _ in range(self.MaxNCS)]
        self.DPEML = [[[0.0] * self.MaxNGS for _ in range(MaxBML)] for _ in range(self.MaxNCS)]
        self.QDCTS = [0.0] * self.MaxCTS
        self.RNMNCS = [[0.0] * MaxQN for _ in range(self.MaxNCS)]
        self.RNLFCS = [0.0] * self.MaxNCS
        self.RNHFCS = [0.0] * self.MaxNCS
        self.RNMC = [0.0] * self.MaxNCS
        self.RNMCCS = [0.0] * self.MaxNCS
        self.RNFPCS = [0.0] * self.MaxNCS
        self.DNBDCS = [0.0] * self.MaxNCS
        self.DHBDCS = [0.0] * self.MaxNCS
        self.ZRNOUT = [0.0] * MaxZQ
        self.QRNOUT = [0.0] * MaxZQ
        self.DQZRN = [0.0] * MaxZQ
        self.NoutCS = [0] * self.MaxNCS
        self.ZTCSmax = [0.0] * self.MaxNCS
        self.TZCSmax = [0.0] * self.MaxNCS
        self.QTCSmax = [0.0] * self.MaxNCS
        self.TQCSmax = [0.0] * self.MaxNCS
        self.STCSmax = [0.0] * self.MaxNCS
        self.TSCSmax = [0.0] * self.MaxNCS
        self.CSHC = [0.0] * self.MaxNCS
        self.CSpm = [0.0] * self.MaxNCS
        self.Dsdt = [0.0] * self.MaxNCS
        self.DAFDT = [0.0] * self.MaxNCS
        self.NoutCS = [0] * self.MaxNCS
        self.ZTCSmax = [0.0] * self.MaxNCS
        self.TZCSmax = [0.0] * self.MaxNCS
        self.QTCSmax = [0.0] * self.MaxNCS
        self.TQCSmax = [0.0] * self.MaxNCS
        self.STCSmax = [0.0] * self.MaxNCS
        self.TSCSmax = [0.0] * self.MaxNCS


def SOL1DFLOWRT(Istep, Imax, Jmax, Grav, Hmin, DTStep, CitaFw, QinT, QLnew, Zout, Qout, DQdZout, Npoint, XXIJ, ZBIJ, DistLG, DX2CS, KNIJ, RNIJ, Zbmin, Qold, Zold, Aold, Bold, DMKold, ALFold):
    Qnew = [0] * Imax
    Znew = [0] * Imax

    for i in range(Imax):
        Zold[i] = CSZW[i]
        Qold[i] = CSQQ[i]

    CompQZBA1(Hmin, DHmin, CitaFw, Imax, Jmax, NPT1CS, KNIJ, RNIJ, DBIJ, ZBIJ, Zbmin, DX2CS, Zold, Aold, Bold, DMKold, ALFold)

    PreissmannScheme(Qold, Zold, Bold, Aold, DMKold, ALFold, Qnew, Znew)

    CompFlowPRMTs(Znew, Qnew)

    for i in range(Imax):
        DAFDT[i] = (CSAA[i] - Aold[i]) / DTstep

    return Qnew, Znew


def PreissmannScheme(QOLD, ZOLD, BOLD, AOLD, DMKOLD, ALFOLD, Qend, Zend):
    QnewT = [0] * Imax
    ZnewT = [0] * Imax
    Qmid = [0] * (Imax - 1)
    Amid = [0] * (Imax - 1)
    Bmid = [0] * (Imax - 1)
    DMKmid = [0] * (Imax - 1)
    ALFmid = [0] * (Imax - 1)
    AddCE = [0] * (Imax - 1)
    AddME = [0] * (Imax - 1)
    AA1 = [0] * (Imax - 1)
    BB1 = [0] * (Imax - 1)
    CC1 = [0] * (Imax - 1)
    DD1 = [0] * (Imax - 1)
    EE1 = [0] * (Imax - 1)
    AA2 = [0] * (Imax - 1)
    BB2 = [0] * (Imax - 1)
    CC2 = [0] * (Imax - 1)
    DD2 = [0] * (Imax - 1)
    EE2 = [0] * (Imax - 1)
    RRR = [0] * Imax
    PPP = [0] * Imax
    Addce = [0] * (Imax - 1)
    Addme = [0] * (Imax - 1)

    Lmax = ITsumF

    for i in range(Imax - 1):
        QLmid[i] = CitaFw * QLNEW[i] + (1.0 - CitaFw) * QLOLD[i]

    for L in range(1, Lmax + 1):
        if L == 1:
            for i in range(Imax):
                QnewT[i] = Qold[i]
                ZnewT[i] = Zold[i]
        elif L > 1:
            for i in range(Imax):
                QnewT[i] = Qnew[i]
                ZnewT[i] = Znew[i]

        for i in range(Imax - 1):
            Qmid[i] = 0.5 * CitaFw * (QnewT[i + 1] + QnewT[i]) + 0.5 * (1.0 - CitaFw) * (Qold[i + 1] + Qold[i])

        CompQZBA2(Zold, ZnewT, Aold, Bold, DMKold, ALFold, DADXmid)

        for i in range(Imax - 1):
            Amid[i] = 0.5 * CitaFw * (Anew[i + 1] + Anew[i]) + 0.5 * (1.0 - CitaFw) * (Aold[i + 1] + Aold[i])
            Bmid[i] = 0.5 * CitaFw * (Bnew[i + 1] + Bnew[i]) + 0.5 * (1.0 - CitaFw) * (Bold[i + 1] + Bold[i])
            DMKmid[i] = 0.5 * CitaFw * (DMknew[i + 1] + DMKnew[i]) + 0.5 * (1.0 - CitaFw) * (DMKold[i + 1] + DMKold[i])
            ALFmid[i] = 0.5 * CitaFw * (ALFnew[i + 1] + ALFnew[i]) + 0.5 * (1.0 - CitaFw) * (ALFold[i + 1] + ALFold[i])

        if NCPFS == 0 or NFLST == 0:
            for i in range(Imax - 1):
                AddCE[i] = 0.0
                AddME[i] = 0.0
        elif NCPFS == 1 and NFLST == 1:
            Dp = Psedi - Pflow
            P0 = (1.0 - Pdry / Psedi) * Pflow + Pdry

            for i in range(Imax - 1):
                pm = (cspm[i] + cspm[i + 1]) / 2.0
                Hcm = (cshc[i] + cshc[i + 1]) / 2.0
                dsdtm = (dSdT[i] + dSdT[i + 1]) / 2.0
                dsdxm = (CSsus[i + 1] - CSSUS[i]) / DX2cs[i]
                DA0dtm = (DASdt1[i] + DASdt1[i + 1]) / 2.0

                Adtm1 = 0.0
                Adtm2 = -(DP) * Qmid[i] * dsdtm / Pm / Psedi
                Adtm3 = -(DP) * (Qmid[i] * Qmid[i] / Amid[i] + Grav * HCm * Amid[i]) * dsdxm / Pm / Psedi

                AddME[i] = Adtm1 + Adtm2 + Adtm3

                AddCE[i] = -DA0dtm

        for i in range(Imax - 1):
            CF1 = Grav * Amid[i] - Alfmid[i] * Bmid[i] * Qmid[i] ** 2.0 / Amid[i] ** 2.0
            CF2 = 2.0 * ALFmid[i] * Qmid[i] / Amid[i]
            CF3 = 2.0 * Dtstep * citafw / Dx2cs[i]

            aa2[i] = -CF1 * CF3
            bb2[i] = 1.0 - CF2 * CF3
            cc2[i] = -aa2[i]
            dd2[i] = 1.0 + CF2 * CF3

            e21 = Qold[i + 1] + Qold[i]
            e22 = -2.0 * Dtstep * CF1 * (1.0 - citafw) * (Zold[i + 1] - Zold[i]) / dx2cs[i]
            e23 = -2.0 * Dtstep * CF2 * (1.0 - Citafw) * (Qold[i + 1] - Qold[i]) / DX2cs[i]
            e24 = (2.0 * dtstep) * 0.0
            e25 = 2.0 * dtstep * ((Qmid[i] / Amid[i]) ** 2.0) * DADXmid[i]
            e26 = -(2.0 * dtstep) * Grav * Amid[i] * abs(Qmid[i]) * Qmid[i] / (DMKmid[i] ** 2.0)

            e27 = -dtstep * Amid[i] * cof * abs(Uup ** 2.0 - Ulw ** 2.0) / Dx2cs[i]

            ee2[i] = e21 + e22 + e23 + e24 + e25 + e26 + e27 + Addme[i] * 2.0 * dtstep

        for i in range(Imax - 1):
            RR1 = -CC2[i] * (aa1[i] + bb1[i] * RRR[i]) + CC1[i] * (aa2[i] + bb2[i] * RRR[i])
            RR2 = dd2[i] * (aa1[i] + bb1[i] * RRR[i]) - dd1[i] * (aa2[i] + bb2[i] * RRR[i])

            RRR[i + 1] = RR1 / RR2

            PP1 = (ee2[i] - bb2[i] * PPP[i]) * (aa1[i] + bb1[i] * RRR[i])
            PP2 = RR2

            PPP[i + 1] = PP1 / PP2

        if KBDout == 1:
            Qnew[Imax - 1] = Qoutnew
            if abs(RRR[Imax - 1]) > 1.0E-6:
                Znew[Imax - 1] = (Qoutnew - PPP[Imax - 1]) / RRR[Imax - 1]
            else:
                print("RRR(imax)=0")
                break

        elif KBDout == 2:
            Znew[Imax - 1] = Zoutnew
            Qnew[Imax - 1] = PPP[Imax - 1] + RRR[Imax - 1] * Zoutnew

        elif KBDout == 3:
            DZNout = (PPP[Imax - 1] - Qold[Imax - 1] + RRR[Imax - 1] * Zold[Imax - 1]) / (DQDZout - RRR[Imax - 1])
            Znew[Imax - 1] = Zold[Imax - 1] + DZNout
            Qnew[Imax - 1] = Qold[Imax - 1] + DQDZout * DZNout
            if Znew[Imax - 1] < CSZBmn[Imax - 1]:
                Znew[Imax - 1] = CSzbmn[Imax - 1] + Dhmin
                Qnew[Imax - 1] = Qmin

        for i in range(Imax - 2, -1, -1):
            DZ11 = (ee2[i] - bb2[i] * PPP[i]) * dd1[i] - (ee1[i] - bb1[i] * PPP[i]) * dd2[i]
            DZ12 = dd1[i] * (aa2[i] + bb2[i] * RRR[i]) - dd2[i] * (aa1[i] + bb1[i] * RRR[i])

            DZ1 = DZ11 / DZ12

            DZ21 = -cc2[i] * dd1[i] + cc1[i] * dd2[i]
            DZ22 = DZ12

            DZ2 = DZ21 / DZ22

            Znew[i] = DZ1 + DZ2 * Znew[i + 1]

            if i != 0:
                Qnew[i] = PPP[i] + RRR[i] * Znew[i]

        errQ = 0.0
        errZ = 0.0
        MDQZ = 1
        for i in range(Imax):
            if MDQZ == 1:
                if abs(Qnew[i]) > 1.0E-3:
                    erq = abs((QnewT[i] - Qnew[i]) / Qnew[i])
                else:
                    erq = 0.0
                Hnew = Znew[i] - CSZBmn[i]
                Hnewt = ZnewT[i] - CSZBmn[i]
                if Hnew > Hmin:
                    erZ = abs((Hnewt - Hnew) / Hnew)
                else:
                    erZ = 0.0
            if MDQZ == 2:
                erq = abs(QnewT[i] - Qnew[i])
                erZ = abs(ZnewT[i] - Znew[i])
            if erq > ErrQ:
                errQ = erq
            if erZ > ErrZ:
                errZ = erz
            if Znew[i] <= CSzbmn[i]:
                Znew[i] = CSzbmn[i] + dhmin
        if errq <= epsQD and errz <= epsZW:
            break
        if L == Lmax:
            break

    for i in range(Imax):
        Qend[i] = Qnew[i]
        Zend[i] = Znew[i]

    for i in range(Imax):
        if Znew[i] < CSzbmn[i]:
            Znew[i] = CSzbmn[i] + dhmin
            Qnew[i] = Qmin

    return Qend, Zend


def CompQZBA1(Hmin, DHmin, CitaFw, Imax, Jmax, NPT1CS, KNIJ, RNIJ, DBIJ, ZBIJ, Zbmin, DX2CS, WLCS, AACS, BBCS, DMKCS, ALFCS):
    for i in range(Imax):
        NumPT = NPT1CS[i]
        for j in range(NumPT):
            if j < NumPT:
                DBJ[j] = DBIJ[i][j]
            ZBJ[j] = ZBIJ[i][j]
            KNJ[j] = KNIJ[i][j]
            RNJ[j] = RNIJ[i][j]

        NMC = NMC1CS[i]
        for k in range(NMC):
            KMCL[k] = NodMCL[i][k]
            KMCR[k] = NodMCR[i][k]

        ZWL = WLCS[i]

        Comparea1(Hmin, DHmin, NumPT, DBJ, ZBJ, KNJ, RNJ, NMC, KMCL, KMCR, ZWL, AAi, BBi, DMKi, alfi)

        AACS[i] = AAi
        BBCS[i] = BBi
        DMKCS[i] = DMKi
        ALFCS[i] = alfi


def CompQZBA2(WLold, WLnew, AACS, BBCS, DMKCS, ALFCS, DADXCS):
    for i in range(Imax):
        NumPT = NPT1CS[i]
        for j in range(NumPT):
            if j < NumPT:
                DBJ[j] = DBIJ[i][j]
            ZBJ[j] = ZBIJ[i][j]
            KNJ[j] = KNIJ[i][j]
            RNJ[j] = RNIJ[i][j]

        NMC = NMC1CS[i]
        for k in range(NMC):
            KMCL[k] = NodMCL[i][k]
            KMCR[k] = NodMCR[i][k]

        ZWL = WLnew[i]

        Comparea1(Hmin, DHmin, NumPT, DBJ, ZBJ, KNJ, RNJ, NMC, KMCL, KMCR, ZWL, AAi, BBi, DMKi, alfi)

        AACS[i] = AAi
        BBCS[i] = BBi
        DMKCS[i] = DMKi
        ALFCS[i] = alfi

    for i in range(Imax - 1):
        Zmd = Zmid[i]

        Comparea2(Hmin, DHmin, NumPT1, DBJ1, ZBJ1, KNJ1, RNJ1, NMC1, KMCL1, KMCR1, Zmd, AACS1)
        Comparea2(Hmin, DHmin, NumPT2, DBJ2, ZBJ2, KNJ2, RNJ2, NMC2, KMCL2, KMCR2, Zmd, AACS2)

        DADXCS[i] = (AACS2 - AACS1) / DX2cs[i]

def Comparea1(Hmin, DHmin, Jmax, DBJ, ZBJ, KNJ, RNJ, NMC, KMCL, KMCR, ZZ, AA, BB, DMK, Alf):
    DBJ = [0.0] * (Jmax - 1)
    ZBJ = [0.0] * Jmax
    HHJ = [0.0] * Jmax
    kNJ = [0.0] * Jmax
    RNJ = [0.0] * Jmax
    KMCL = [0.0] * NMC
    KMCR = [0.0] * NMC

    MODCSZW(Hmin, JMAX, NMC, KmcL, KmcR, KNJ, ZBJ, HHJ, ZZ)

    BB = 0.0
    AA = 0.0
    DMk = 0.0
    ALF1 = 0.0
    for J in range(1, Jmax - 1):
        CMAN = (RNJ[J] + RNJ[J + 1]) / 2.0
        DX = DBJ[J]

        if (HHJ[J] > Hmin and HHJ[J + 1] > Hmin):
            DB = DX
            DA = (DB) * (HHJ[J] + HHJ[J + 1]) * 0.5

            WP = (DB ** 2.0 + (HHJ[J] - HHJ[J + 1]) ** 2.0) ** 0.5
            RD = DA / WP
        if (HHJ[J] <= Hmin and HHJ[J + 1] <= Hmin):
            DB = 0.0
            DA = 0.0
            WP = 0.0
            RD = 0.0
        if (HHJ[J] <= Hmin and HHJ[J + 1] > Hmin):
            HJ1 = HHJ[J + 1]
            HJ = ZBJ[J] - ZBJ[J + 1] - HJ1
            if (HJ >= 0.0):
                DB = (HJ1 / (HJ1 + HJ)) * DX
            else:
                DB = DX
            DA = DB * HJ1 * 0.5

            WP = (DB ** 2.0 + HJ1 ** 2.0) ** 0.5
            RD = DA / WP
        if (HHJ[J] > Hmin and HHJ[J + 1] <= Hmin):
            HJ = HHJ[J]
            HJ1 = ZBJ[J + 1] - ZBJ[J] - HJ

            if (HJ1 >= 0.0):
                DB = (HJ / (HJ + HJ1)) * DX
            else:
                DB = DX
            DA = DB * HJ * 0.5

            WP = (DB ** 2.0 + HJ ** 2.0) ** 0.5
            RD = DA / WP

        if (RD > 0.0):
            if ((KNJ[J] == 0 or KNJ[J] == 3) and DA / RD <= DHmin):
                Cman = 0.020
            else:
                Cman = Cman
            DK = (DA / CMAN) * ((RD) ** (2.0 / 3.0))
        else:
            Dk = 0.0

        if (DA > 0.0):
            alf1 = alf1 + da * (dk / da) ** 2.0
            alf1 = alf1 + (DK ** 3.0) / (DA ** 2.0)
        else:
            alf1 = alf1 + 0.0

        AA = AA + DA
        BB = DB + BB
        DMk = DMk + DK

    if (AA > 0.0):
        alf2 = (dmk ** 3.0) / (AA ** 2.0)
        Alf = alf1 / alf2
    else:
        Alf = 1.0

def Comparea2(Hmin, DHmin, Jmax, DBJ, ZBJ, KNJ, RNJ, NMC, KMCL, KMCR, ZZ, AA):
    DBJ = [0.0] * (Jmax - 1)
    ZBJ = [0.0] * Jmax
    HHJ = [0.0] * Jmax
    kNJ = [0.0] * Jmax
    RNJ = [0.0] * Jmax
    KMCL = [0.0] * NMC
    KMCR = [0.0] * NMC

    MODCSZW(Hmin, JMAX, NMC, KmcL, KmcR, KNJ, ZBJ, HHJ, ZZ)

    AA = 0.0
    for J in range(1, Jmax - 1):
        CMAN = (RNJ[J] + RNJ[J + 1]) / 2.0
        DX = DBJ[J]

        if (HHJ[J] > Hmin and HHJ[J + 1] > Hmin):
            DB = DX
            DA = (DB) * (HHJ[J] + HHJ[J + 1]) * 0.5
        if (HHJ[J] <= Hmin and HHJ[J + 1] <= Hmin):
            DB = 0.0
            DA = 0.0
        if (HHJ[J] <= Hmin and HHJ[J + 1] > Hmin):
            HJ1 = HHJ[J + 1]
            HJ = ZBJ[J] - ZBJ[J + 1] - HJ1
            if (HJ >= 0.0):
                DB = (HJ1 / (HJ1 + HJ)) * DX
            else:
                DB = DX
            DA = Db * HJ1 * 0.5
        if (HHJ[J] > Hmin and HHJ[J + 1] <= Hmin):
            HJ = HHJ[J]
            HJ1 = ZBJ[J + 1] - ZBJ[J] - HJ

            if (HJ1 >= 0.0):
                DB = (HJ / (HJ + HJ1)) * DX
            else:
                DB = DX
            DA = DB * HJ * 0.5

        AA = AA + DA

    return AA

def CompFlowPRMTs(ZCS, QCS):
    QCS = [0.0] * self.MaxNCS
    ZCS = [0.0] * self.MaxNCS

    DBJ = [0.0] * MaxNPT
    ZBJ = [0.0] * MaxNPT
    HHJ = [0.0] * MaxNPT
    kNJ = [0.0] * MaxNPT
    RNJ = [0.0] * MaxNPT
    KMCL = [0.0] * self.MaxNMC
    KMCR = [0.0] * self.MaxNMC

    MODCSZW(HK, JMAX, NMC, KmcL, KmcR, KC, ZB, HH, CSZW)

    for I in range(1, Imax + 1):
        CSZW[I] = ZCS[i]
        CSQQ[I] = QCS[I]

    if (MDRGH == 1):
        COMPRNofCHFP1()

    if (Istep == 1 and SPtime < 2.0 * Dtstep):
        COMPRNofCHFP1()

    for I in range(1, Imax + 1):
        for J in range(1, NPT1CS[I] + 1):
            if (J != NPT1CS[I]):
                DBJ[J] = DBIJ[I][J]
            ZBJ[J] = ZBIJ[I][J]
            RNJ[J] = RNIJ[I][J]
            KCJ[J] = KNIJ[I][J]

        NMC = NMC1CS[i]
        for K in range(1, NMC + 1):
            KMCL[k] = NodMCL[i][k]
            KMCR[k] = NodMCR[i][k]

        COMParea3(I, Hmin, GRAV, NPT1cs(I), Nmc, KMCL, KMCR, ZBJ, RNJ, KCJ, DBJ, AAJ, BBJ, HHJ, QKJ, UUJ, UTJ, WPJ,
                  RDJ, QQJ, Hnd, Und, Qnd, TFnd, UTnd, CSZW(I), CSQQ(I), CSUU(I), CSHH(I), CSAA(I), CSBB(i),
                  CSQm(I), CSUm(I), CSHm(I), CSAm(I), CSBm(I), CSQf(I), CSUf(I), CSHf(I), CSAf(I), CSBf(I), CSZB(I),
                  CSSP(I), CSQK(I), CSRN(I), CSUT(I), CSWP(I), CSRD(I), CSFR(I), CSHC(i))

    for I in range(1, Imax + 1):
        for J in range(1, NPT1cs(I) + 1):
            AAIJ[I][J] = AAJ[J]
            BBIJ[I][J] = BBJ[J]
            HHIJ[I][J] = HHJ[J]
            QKIJ[I][J] = QKJ[J]
            UUIJ[I][J] = UUJ[J]
            UTIJ[I][J] = UTJ[J]
            QqIJ[I][J] = QQJ[J]
            WPIJ[I][J] = WPJ[J]
            RDIJ[I][J] = RDJ[J]

            Hnode[I][J] = Hnd[J]
            Unode[I][J] = Und[J]
            Qnode[I][J] = Qnd[J]
            UTnode[I][J] = UTnd[J]
            TFnode[I][J] = TFnd[J]

    if (CSBB > 0.0):
        CSHH = CSAA / CSBB
        CSRD = CSAA / CSWP
        CSHC = SumAHc / CSAA
    else:
        CSHH = 0.0
        CSRD = 0.0
        CSHC = 0.0

    CSZB = CSZW - CSHH

    SUMKA = 0.0
    for J in range(1, Jmax - 1):
        if (AAJ[J] > 0.0):
            SUMKA = SUMKA + QKJ[J] ** 3.0 / AAJ[J] ** 2.0
        else:
            SUMKA = SUMKA + 0.0

    if (CSAA > 0.0):
        CSUU = CSQQ / CSAA
        CSSP = CSQQ ** 2.0 / CSQK ** 2.0
        CSRN = (CSRD ** (2.0 / 3.0)) * SQRT(CSSP) / CSUU
    else:
        CSUU = 0.0
        CSSP = 0.0
        CSRN = 0.02

    CSUT = SQRT(GRAV) * CSRN * CSUU / (CSRD ** (1.0 / 6.0))

    SumQm = 0.0
    SumAm = 0.0
    SumBm = 0.0

    SumQf = 0.0
    SumAf = 0.0
    SumBf = 0.0

    for J in range(1, Jmax - 1):
        if (HHJ[J] > 0.0):
            UUJ[J] = (RDJ[J] ** (2.0 / 3.0)) * (CSSP ** 0.5) / RNJ[J]
            UTJ[J] = SQRT(GRAV) * RNJ[J] * UUJ[J] / (RDJ[J] ** (1.0 / 6.0))
        else:
            UUJ[J] = 0.0
            UTJ[J] = 0.0
        QQJ[J] = UUJ[J] * HHJ[J]
        QQJ[J] = CSQQ * (QKJ[J] / CSQK)
        if (KC[J] * KC[J + 1] != 2 and KC[J] * KC[J + 1] != 4):
            SumQM = SumQM + QQJ[J]
            SumAm = SumAm + AAJ[J]
            SumBm = SumBm + BBJ[J]
        if (KC[J] * KC[J + 1] != 0):
            SumQf = SumQf + QQJ[J]
            SumAf = SumAf + AAJ[J]
            SumBf = SumBf + BBJ[J]

    CSQm = SumQm
    CSAm = SumAm
    CSBm = SumBm

    CSQf = SumQf
    CSAf = SumAf
    CSBf = SumBf

    if (SumBm > 0.0):
        CSHm = SumAm / SumBm
    else:
        CSHm = 0.0

    if (SumBf > 0.0):
        CSHf = SumAf / SumBf
    else:
        CSHf = 0.0

    if (CSAm > 0.0):
        CSUm = CSQm / CSAm
    else:
        CSUm = 0.0

    if (CSAf > 0.0):
        CSUf = CSQf / CSAf
    else:
        CSUf = 0.0

    for J in range(1, Jmax):
        Hnd[J] = HH[J]
        if (Hnd[J] > HK):
            Und[J] = (Hnd[J] ** (2.0 / 3.0)) * (CSSP ** 0.5) / RN[J]
            Qnd[J] = Hnd[J] * Und[J]
            UTnd[J] = SQRT(GRAV) * RN[J] * Und[J] / (Hnd[J] ** (1.0 / 6.0))
            TFnd[J] = (UTnd[J] ** 2.0) * 1000.0
        else:
            Und[J] = 0.0
            Qnd[J] = 0.0
            UTnd[J] = 0.0
            TFnd[J] = 0.0

    if (CSAA > 0.0):
        CSALF = SUMKA / (CSQK ** 3.0 / CSAA ** 2.0)
    else:
        CSALF = 1.0

    if (CSAA > 0.0):
        CSFR = (CSUU) * sqrt(CSALF) / SQRT(9.81 * CSHH)
    else:
        CSFR = 1.01




