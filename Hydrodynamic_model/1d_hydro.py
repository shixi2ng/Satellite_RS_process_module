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
        self.Nstart = 0  # 初始文件输入格式(1=有初始文件,0=无初始文件)
        self.IMax = 0  # Imax :断面数,Jmax:断面上最大节点数
        self.JMax = 0
        self.NumTBY = 0  # 支流数量
        self.NumGS = 0  # NumGS:非均匀泥沙分组数 Number of Graded Sediments
        self.NCTCS = 0  # 计算河段内控制断面数量(1=inlet,Nctcs=outlet)
        self.NFLDiv = 0  # 引水引沙参数=0(不考虑),
        self.NTRD = 0  # 记录结果的时间步数
        self.NCPFS = 0  # 水沙是否耦合计算
        self.MDRGH = 0  # 糙率计算方法 (1)=Q-N (2)Zhang EQ (3)Other EQ.
        self.NYspin = 0  # Spin-up period(0=不计算/1=计算) Flag for Spin-up
        self.NFLST = 0  # 泥沙输移是否计算(0=不计算/1=计算) Flag for Sediment Transport
        self.NFLBD = 0  # 床面变形是否计算(0=不计算/1=计算) Flag for bed deformation
        self.NFLGA = 0  # 计算过程中床沙级配是否调整 (0=不计算/ 1=计算) Flag for Gradation Adjustment.
        self.ITSUMF = 0  # 水流计算中的最大迭代次数
        self.ITSumS = 0  # 含沙量计算中的迭代次数 Lmax>=2
        self.MaxNML = 0  # 最大/最小/初始 床沙记忆层的层数
        self.MinNML = 0
        self.NMLini = 0
        self.MDLDA = 0  # 冲淤面积横向分配方法
        self.MDScc = 0  # 挟沙力计算方法选择
        self.KBDout = 0  # 下游边界类型
        self.NumTQint = 0  # 进口Time-Qint过程的个数
        self.NumTMP = 0  # 进口Time-Wtemp过程的个数
        self.NumTQout = 0  # 出口Time-Qout过程的个数
        self.NumTZout = 0  # 出口Time-Zout过程的个数
        self.NumZQout = 0  # 出口水位流量关系的个数
        self.NumQN = 0  # 实际QN关系的个数
        self.NumTSint = 0  # 主流进口断面T-S个数
        self.NumTdiv = 0  # 沿程Time-Q过程的个数
        self.NumTsdiv = 0  # 沿程Time-S过程的个数 !ZXLPR新加入的 原来只有NumTdiv, !沿程Time-Q/S过程的个数
        self.Istep = 0  # 第ISTEP时间步
        self.Nstep = 0  # 总计算步数
        self.NumOCS = 0  # 要输出结果的断面数量
        self.MaxNCS = 0  # 最大断面数
        self.MaxNPT = 0  # 最大节点数
        self.MaxNMC = 0  # 断面内的最大主槽划分数 (MaxNMC<=3)
        self.MaxTBY = 0  # 最大支流数
        self.MaxCTS = 0  # 最大控制断面数
        self.MaxStp = 0  # 最大总计算时段数(Definition)
        self.MaxNGS = 0  # 最大泥沙分组数
        self.MaxQN = 0  # 最大数量 (流量与糙率的关系曲线)
        self.MaxZQ = 0  # 最大数量 (出口水位流量关系曲线)
        self.TimeSM = 0.0  # 总的模拟时间
        self.TimeSP = 0.0  # 设置好的Spin-up time (hours)
        self.SPTime = 0.0  # 计算中的Spin-up time (hours)
        self.DTstep = 0.0  # 固定时间步长
        self.TimeSC = 0.0  # 已计算时间(Second)
        self.TimeHR = 0.0  # 已计算时间(Hour  )
        self.Hmin = 0.0  # 干湿控制水深(一般取0.10m)
        self.Qmin = 0.0  # 计算中的最小流量
        self.DHmin = 0.0  # 干湿控制水深(一般取0.10m)
        self.Coks = 0.0
        self.Coms = 0.0
        self.VistFW = 0.0
        self.CitaDpS = 0.0
        self.EPSQD = 0.0
        self.EPSZW = 0.0
        self.CitaFW = 0.0
        self.DQDZout = 0.0
        self.Qintnew = 0.0
        self.Zoutnew = 0.0
        self.Qoutnew = 0.0
        self.GRAV = 0.0
        self.PSedi = 0.0
        self.PFlow = 0.0
        self.Pdry = 0.0
        self.ZWMVL = 0.0
        self.ERRSD = 0.0
        self.CitaSD = 0.0
        self.DMix = 0.0
        self.Dmem = 0.0
        self.DZmor = 0.0
        self.THmin = 0.0
        self.CofAdeg = 0.0
        self.CofBdeg = 0.0
        self.CofAdep = 0.0
        self.CofBdep = 0.0
        self.Wtemp = 0.0


        self.NPT1CS = [0] * MaxNCS
        self.NMC1CS = [0] * MaxNCS
        self.NodMCL = [[0] * MaxNMC for _ in range(MaxNCS)]
        self.NodMCR = [[0] * MaxNMC for _ in range(MaxNCS)]
        self.IPCTCS = [0] * MaxCTS
        self.NCSTBY = [0] * MaxTBY
        self.NTPML = [0] * MaxNcs
        self.NumQtby = [0] * MaxStp
        self.NumStby = [0] * MaxStp
        self.DISTLG = [0.0] * MaxNcs
        self.Dx2cs = [0.0] * MaxNcs
        self.DistCTS = [0.0] * MaxCTS
        self.DM1FRT = [0.0] * MaxNGS
        self.DMSiev = [0.0] * MaxNGS
        self.TimeTQ = [0.0] * MaxStp
        self.TimeQint = [0.0] * MaxStp
        self.TimeTS = [0.0] * MaxStp
        self.TimeSint = [0.0] * MaxStp
        self.TimeTout = [0.0] * MaxStp
        self.TimeZout = [0.0] * MaxStp
        self.TimeQout = [0.0] * MaxStp
        self.TimeSkint = [[0.0] * MaxNGS for _ in range(MaxStp)]
        self.TimeTMP = [0.0] * MaxStp
        self.TimeWtemp = [0.0] * MaxStp
        self.TimeTdiv = [0.0] * MaxStp
        self.TimeTSdiv = [0.0] * MaxStp
        self.TimeWdiv = [[0.0] * MaxNcs for _ in range(MaxStp)]
        self.TimeSdiv = [[0.0] * MaxNcs for _ in range(MaxStp)]
        self.QLold = [0.0] * MaxNcs
        self.QLnew = [0.0] * MaxNcs
        self.TimeTStby = [[0.0] * MaxStp for _ in range(MaxTBY)]
        self.TimeStby = [[0.0] * MaxStp for _ in range(MaxTBY)]
        self.TimeSKtby = [[[0.0] * MaxNGS for _ in range(MaxStp)] for _ in range(MaxTBY)]
        self.TimeQSKtby = [[[0.0] * MaxNGS for _ in range(MaxStp)] for _ in range(MaxTBY)]
        self.TimeTQtby = [[0.0] * MaxStp for _ in range(MaxTBY)]
        self.TimeQtby = [[0.0] * MaxStp for _ in range(MaxTBY)]
        self.CSZW = [0.0] * MaxNcs
        self.CSQQ = [0.0] * MaxNcs
        self.CSUU = [0.0] * MaxNcs
        self.CSHH = [0.0] * MaxNcs
        self.CSAA = [0.0] * MaxNcs
        self.CSBB = [0.0] * MaxNcs
        self.CSZB = [0.0] * MaxNcs
        self.CSSP = [0.0] * MaxNcs
        self.CSRN = [0.0] * MaxNcs
        self.CSUT = [0.0] * MaxNcs
        self.CSWP = [0.0] * MaxNcs
        self.CSRD = [0.0] * MaxNcs
        self.CSFR = [0.0] * MaxNcs
        self.CSQK = [0.0] * MaxNcs
        self.CSQM = [0.0] * MaxNcs
        self.CSUM = [0.0] * MaxNcs
        self.CSHM = [0.0] * MaxNcs
        self.CSAM = [0.0] * MaxNcs
        self.CSBM = [0.0] * MaxNcs
        self.CSQf = [0.0] * MaxNcs
        self.CSUf = [0.0] * MaxNcs
        self.CSHf = [0.0] * MaxNcs
        self.CSAf = [0.0] * MaxNcs
        self.CSBf = [0.0] * MaxNcs
        self.CSZBmn = [0.0] * MaxNcs
        self.CSZBav = [0.0] * MaxNcs
        self.CSZBav0 = [0.0] * MaxNcs
        self.CSSUS = [0.0] * MaxNcs
        self.CSSCC = [0.0] * MaxNcs
        self.SUSD50 = [0.0] * MaxNcs
        self.SUSDPJ = [0.0] * MaxNcs
        self.CSSVL = [0.0] * MaxNcs
        self.CSSVM = [0.0] * MaxNcs
        self.CSCNK = [0.0] * MaxNcs
        self.WSmed = [0.0] * MaxNcs
        self.DBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DKIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.AAIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.QKIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.QQIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.UUij = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.BBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.HHij = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.UTIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.WPIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.RDIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.XXIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.ZBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.CSZW = [0.0] * MaxNcs
        self.CSQQ = [0.0] * MaxNcs
        self.CSUU = [0.0] * MaxNcs
        self.CSHH = [0.0] * MaxNcs
        self.CSAA = [0.0] * MaxNcs
        self.CSBB = [0.0] * MaxNcs
        self.CSZB = [0.0] * MaxNcs
        self.CSSP = [0.0] * MaxNcs
        self.CSRN = [0.0] * MaxNcs
        self.CSUT = [0.0] * MaxNcs
        self.CSWP = [0.0] * MaxNcs
        self.CSRD = [0.0] * MaxNcs
        self.CSFR = [0.0] * MaxNcs
        self.CSQK = [0.0] * MaxNcs
        self.CSQM = [0.0] * MaxNcs
        self.CSUM = [0.0] * MaxNcs
        self.CSHM = [0.0] * MaxNcs
        self.CSAM = [0.0] * MaxNcs
        self.CSBM = [0.0] * MaxNcs
        self.CSQf = [0.0] * MaxNcs
        self.CSUf = [0.0] * MaxNcs
        self.CSHf = [0.0] * MaxNcs
        self.CSAf = [0.0] * MaxNcs
        self.CSBf = [0.0] * MaxNcs
        self.CSZBmn = [0.0] * MaxNcs
        self.CSZBav = [0.0] * MaxNcs
        self.CSZBav0 = [0.0] * MaxNcs
        self.CSSUS = [0.0] * MaxNcs
        self.CSSCC = [0.0] * MaxNcs
        self.SUSD50 = [0.0] * MaxNcs
        self.SUSDPJ = [0.0] * MaxNcs
        self.CSSVL = [0.0] * MaxNcs
        self.CSSVM = [0.0] * MaxNcs
        self.CSCNK = [0.0] * MaxNcs
        self.WSmed = [0.0] * MaxNcs
        self.DBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DKIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.AAIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.QKIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.QQIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.UUij = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.BBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.HHij = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.UTIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.WPIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.RDIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.XXIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.ZBIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DFCIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.TCnode = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.COKSD = [0.0] * MaxNcs
        self.COMSD = [0.0] * MaxNcs
        self.WSET = [0.0] * MaxNcs
        self.WSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.UCIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.SKint = [0.0] * MaxStp
        self.ALFSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.BLTSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.SUSIK1 = [0.0] * MaxNcs
        self.SUSIK2 = [0.0] * MaxNcs
        self.SUSIK3 = [0.0] * MaxNcs
        self.SUSIK4 = [0.0] * MaxNcs
        self.SUSIK5 = [0.0] * MaxNcs
        self.SUSIK6 = [0.0] * MaxNcs
        self.SUSIK7 = [0.0] * MaxNcs
        self.SUSIK8 = [0.0] * MaxNcs
        self.SUSIK9 = [0.0] * MaxNcs
        self.SUSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.SCCIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.DPSCCIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.DPSUSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.QSLold = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.QSLnew = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.CSASIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.CSBSIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.BED50 = [0.0] * MaxNcs
        self.BED90 = [0.0] * MaxNcs
        self.BedPJ = [0.0] * MaxNcs
        self.BED10 = [0.0] * MaxNcs
        self.BED60 = [0.0] * MaxNcs
        self.CSSL1 = [0.0] * MaxNcs
        self.CSSL2 = [[0.0] * MaxTBY for _ in range(MaxNcs)]
        self.CSDA = [0.0] * MaxNcs
        self.DAMC = [0.0] * MaxNcs
        self.DAIJ = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DZIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.DAIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.DHnode = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DZnode = [[0.0] * MaxNpt for _ in range(MaxNcs)]
        self.DZMC = [0.0] * MaxNcs
        self.dASdt1 = [0.0] * MaxNcs
        self.dASdt2 = [0.0] * MaxNcs
        self.PDASIK1 = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.PDASIK2 = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.DASIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.VOLCSold = [0.0] * (MaxNcs - 1)
        self.VOLMCold = [0.0] * (MaxNcs - 1)
        self.TimeVLcs = [0.0] * MaxNcs
        self.TimeVLMC = [0.0] * MaxNcs
        self.TimeVL2CTCS = [[0.0] * MaxCTS for _ in range(MaxNcs)]
        self.TimeVL2CTMC = [[0.0] * MaxCTS for _ in range(MaxNcs)]
        self.CofNDX = [0.0] * MaxNcs
        self.CofNDY = [0.0] * MaxNcs
        self.DPBEDIK = [[0.0] * MaxNGS for _ in range(MaxNcs)]
        self.THBAL = [0.0] * MaxNcs
        self.THEML = [[0.0] * MaxBML for _ in range(MaxNcs)]
        self.DPEML = [[[0.0] * MaxNGS for _ in range(MaxBML)] for _ in range(MaxNcs)]
        self.QDCTS = [0.0] * MaxCTS
        self.RNMNCS = [[0.0] * MaxQN for _ in range(MaxNcs)]
        self.RNLFCS = [0.0] * MaxNcs
        self.RNHFCS = [0.0] * MaxNcs
        self.RNMC = [0.0] * MaxNcs
        self.RNMCCS = [0.0] * MaxNcs
        self.RNFPCS = [0.0] * MaxNcs
        self.DNBDCS = [0.0] * MaxNcs
        self.DHBDCS = [0.0] * MaxNcs
        self.ZRNOUT = [0.0] * MaxZQ
        self.QRNOUT = [0.0] * MaxZQ
        self.DQZRN = [0.0] * MaxZQ
        self.NoutCS = [0] * MaxNcs
        self.ZTCSmax = [0.0] * MaxNcs
        self.TZCSmax = [0.0] * MaxNcs
        self.QTCSmax = [0.0] * MaxNcs
        self.TQCSmax = [0.0] * MaxNcs
        self.STCSmax = [0.0] * MaxNcs
        self.TSCSmax = [0.0] * MaxNcs
        self.CSHC = [0.0] * MaxNcs
        self.CSpm = [0.0] * MaxNcs
        self.Dsdt = [0.0] * MaxNcs
        self.DAFDT = [0.0] * MaxNcs
        self.NoutCS = [0] * MaxNcs
        self.ZTCSmax = [0.0] * MaxNcs
        self.TZCSmax = [0.0] * MaxNcs
        self.QTCSmax = [0.0] * MaxNcs
        self.TQCSmax = [0.0] * MaxNcs
        self.STCSmax = [0.0] * MaxNcs
        self.TSCSmax = [0.0] * MaxNcs


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
    QCS = [0.0] * MaxNcs
    ZCS = [0.0] * MaxNCS

    DBJ = [0.0] * MaxNPT
    ZBJ = [0.0] * MaxNPT
    HHJ = [0.0] * MaxNPT
    kNJ = [0.0] * MaxNPT
    RNJ = [0.0] * MaxNPT
    KMCL = [0.0] * MaxNMC
    KMCR = [0.0] * MaxNMC

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




