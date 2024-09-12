#############################################################################################
# 1D Morphodynamic Model for the Yellow River  using a coupled approach
#                         Developed by Junqiang Xia
#     						   May-July 2010
#  Characteristics of this model:
#  (1) Quasi-2D or 1D,Different Roughness between Main Channel and Flood Plain
#  (2) Nonequlibrium transport of nonuniform Suspended Sediments
#  (3) Adapt to the Low and High Sediment Concentrations
#  (4) unsteady Flow and coupled solution
#############################################################################################
import copy
import shutil
import time
import traceback

import numpy as np
import os
import pandas as pd
import basic_function as bf


class HydrodynamicModel_1D(object):

    def __init__(self, work_env: str):

        # Define PATH
        if not os.path.exists(work_env):
            try:
                os.makedirs(work_env, exist_ok=True)
                self.work_env = work_env
            except:
                print(traceback.format_exc())
                raise Exception('Need a valid work environment')
        else:
            self.work_env = work_env

        # Define the input and output path
        self.input_path = None
        self.output_path = None

        # User-defined para
        self.ROI_name: str = ''

        # Define maximum number
        self.MaxNCS = 100  # Maximum number of cross-sections
        self.MaxNGS = 10  # Maximum number of sediment groups
        self.MaxStp = 1000  # Maximum number of time steps
        self.MaxCTS = 100  # Maximum number of control cross-sections
        self.MaxTBY = 10  # Maximum number of tributaries
        self.MaxQN = 10  # Maximum number of flow rates for roughness calculation

        self.MaxNPT = 0  # 最大节点数
        self.MaxNMC = 0  # 断面内的最大主槽划分数 (MaxNMC<=3)
        self.MaxNGS = 0  # 最大泥沙分组数
        self.MaxQN = 0  # 最大数量 (流量与糙率的关系曲线)
        self.MaxZQ = 0  # 最大数量 (出口水位流量关系曲线)

        # Define the global parameters
        self.Imax: int = 0  # 计算断面数
        self.Jmax: int = 0  # 断面上最大节点数
        self.TimeSM: float = 0.0  # 模拟时间(hour)
        self.NYspin: bool = True
        self.TimeSP: float = 0.0  # 设置好的Spin-up time (hours)
        self.SPTime: float = 0.0  # 计算中的Spin-up time (hours)
        self.TimeSC = 0.0  # 已计算时间(Second)
        self.TimeHR = 0.0  # 已计算时间(Hour)
        self.DTstep: int = 30  # 计算固定的时间步长(Unit: Second) 45 不可算
        self.NTRD: int = 480  # 记录计算结果的时间步数
        self.Num_TBY: int = 0  # 实际支流数量
        self.Num_Control_CS: int = 0  # 控制断面数量
        self.Num_GS: int = 12  # 非均匀沙分组数
        self.NCPFS: bool = True  # 水沙是否耦合计算(False = 非耦合; True = 耦合)
        self.MDRGH: int = 4  # 糙率计算方法(1 = Q - n; 2 = ZhangHW Eq; 3 = Other Eq; 4 = liu Xin Eq)主槽区域
        self.Nstart: bool = True  # 初始文件(False = 无初始文件; True = 有初始文件)
        self.NFLDiv: bool = False  # 引水引沙参数=0(不考虑),

        # Define the flow parameters
        self.Grav: float = 9.81  # 重力加速度
        self.CitaFW: float = 0.75  # Pressiman Scheme中的时间步参数 0.75 0.65
        self.Hmin: float = 0.01  # 计算最小水深(m)
        self.DHmin: float = 0.2  # 计算最小水深(m)
        self.Qmin: float = 1.00  # 计算中的最小流量
        self.ITSUMF: int = 100  # 水流计算中的最大迭代次数 50 or 100
        self.EPSQD: float = 0.005  # 水流计算Discharge迭代误差
        self.EPSZW: float = 0.005  # 水流计算中的水位迭代误差

        # Flow parameter defined during the outlet import
        self.KBDout = None  # 下游边界类型
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
        self.NumOCS = 0  # 要输出结果的断面数量
        self.Wtemp = 0.0
        self.DQDZout = 0.0
        self.Qintnew = 0.0
        self.Zoutnew = 0.0
        self.Qoutnew = 0.0
        self.VistFW = 0.0

        # Define the sediment parameters
        self.NFLST: bool = True  # 泥沙输移是否计算(True = 不计算 / False = 计算)  Flag for Sediment Transport
        self.NFLBD: bool = True  # 河床冲淤是否计算(True = 不计算 / False = 计算)  Flag for bed deformation
        self.NFLGA: bool = True  # 计算过程中床沙级配是否调整(True = 不计算 / False = 计算) Flag for Gradation Adjustment
        self.Pflow: float = 1000.0  # 水流密度(1000 kg / m ^ 3)
        self.Psedi: float = 2650.0  # 泥沙密度(2650 kg / m ^ 3)
        self.Pdry: float = 1400.0  # 床沙干密度(1400 kg / m ^ 3)
        self.ZWMVL: float = 100.0  # 河段内的最高水位, 用于断面法计算冲淤
        self.ERRSD: float = 0.10  # 含沙量计算中的误差控制
        self.CitaSD: float = 0.50  # 泥沙控制方程离散权重系数，一般取0.5 0.6
        self.CokZ: float = 2.500  # 张红武挟沙力公式的系数
        self.ComZ: float = 0.620  # 张红武挟沙力公式的系数
        self.CokW: float = 0.452  # 吴保生挟沙力公式的系数
        self.ComW: float = 0.762  # 吴保生挟沙力公式的系数
        self.CokZRJ: float = 0.05  # 张瑞瑾挟沙力公式的系数
        self.ComZRJ: float = 1.55  # 张瑞瑾挟沙力公式的系数
        self.CokV: float = 1.0  # VanRijn 挟沙力公式的系数k,m
        self.ComV: float = 1.0  # VanRijn 挟沙力公式的系数k,m
        self.Cokm = [(self.CokZ, self.ComZ), (self.CokW, self.ComW), (self.CokZRJ, self.ComZRJ), (self.CokV, self.ComV)]
        self.MDSCC: int = 2  # 挟沙力计算方法选择(1 = ZHangHW; 2 = WuBS; 3 = ZhangRJ; 4 = Van Rijn)
        self.CitaDPS: float = 0.75  # 挟沙力级配与悬沙级配考虑 DPSCCK(N) = CITA * DPSCCK(N) + (1.0 - CITA) * DPSUSK(N) 0.70
        self.COFaDEG: float = 0.001  # =0.001/(ws)^0.70   冲刷时系数!DEG=Degradation alf=A/(w)^B   0.8
        self.COFbDEG: float = 0.7  # =0.001/(ws)^0.70   冲刷时系数!DEG=Degradation alf=A/(w)^B   0.8
        self.COFaDEP: float = 0.001  # =0.001/(ws)^0.30   淤积时系数!DEP=Deposition  0.2
        self.COFbDEP: float = 0.3  # =0.001/(ws)^0.30   淤积时系数!DEP=Deposition  0.2
        self.MDLDA: int = 2  # 冲淤面积横向分配方法(1 = 等厚淤积分布, 主槽冲刷
                             #  2 = 按子断面流量分配冲淤面积 #  3 = 按过水面积大小分配冲淤面积
                             #  4 = 按挟沙能力大小分配冲淤面积 #  5 = 按水流切应力大小分配冲淤面积
        self.MaxNML: int = 20  # 最大床沙记忆层的层数
        self.MinNML: int = 6  # 最小床沙记忆层的层数
        self.NMLini: int = 12  # 初始床沙记忆层的层数
        self.ITSumS: int = 0
        self.Dmix: float = 2.0  # 混合层厚度
        self.Dmem: float = 0.5  # 各层记忆层厚度
        self.DZmor: float = 4.0  # 最大冲淤幅度
        self.THmin: float = 0.00001  # 床沙级配调整计算的临界冲淤厚

        # Cross-section profile list / arr
        self.CS_name = []  # Cross-section name
        self.CS_DistViaRiv = []  # Distance along river for each cross-section
        self.CS_node_num = []  # Node number for each cross-section
        self.XXIJ = []  # Distance to the left bank for each node on each cross-section
        self.ZBIJ = []  # Elevation for each node on each cross-section
        self.KNIJ = []  # Channel or floodplain indicator for each node on each cross-section
        self.ZBINL = []  # Elevation for each node on each cross-section (Initial value)
        self.NMC1CS = []  # Number of channel for each cross-section
        self.NodMCL = []  # Node of left bank for each channel of each cross-section主槽左侧滩地节点号
        self.NodMCR = []  # Node of right bank for each channel of each cross-section 主槽右侧滩地节点号
        self.BWMC = []  # Width of main channel for each cross-section
        self.ZBBF = []  # Bankfull elevation for each cross-section
        self.DX2CS = []  # Distance between two cross-sections
        self.CSZBmn = []  # The lowest elevation (Thalweg) for each cross-section
        self.CSZBav = []  # The average elevation of main channel
        self.CSZBav0 = []  # Initial record of the average elevation of main channel

        # Inform for sub-cross sections
        self.DBIJ = []  # 各子断面宽度
        self.DKIJ = []  # 各子断面代号(0*1)(0*2)均为主槽
        self.AAIJ = []  # 各子断面的过水面积
        self.QKIJ = []  # 各子断面的流量模数
        self.QQIJ = []  # 各子断面的流量
        self.UUij = []  # 各子断面的流速
        self.BBIJ = []  # 各子断面过水的水面宽度
        self.HHij = []  # 各子断面过水的水面水深
        self.UTIJ = []
        self.WPIJ = []  # 各子断面的湿周
        self.RDIJ = []  # 各子断面的水力半径

        # Control cross-section profile
        self.Control_CS_Id = []  # ID for each control cross-section
        self.Control_CS_Name = [] # Name for each control cross-section
        self.Control_CS_DistViaRiv = []  # Distance along river for each control cross-section

        # Bed Material Gradation
        self.DMSiev = np.zeros(self.MaxNGS)  # 筛分粒径
        self.NoCSsp = np.zeros(self.MaxNCS, dtype=int)  # Cross-sections with bed gradation
        self.DistSP = np.zeros(self.MaxNCS)  # Distance of cross-sections with bed gradation
        self.PBEDSP = np.zeros((self.MaxNCS, self.MaxNGS))  # Bed gradation at cross-sections
        self.PKtemp = np.zeros(self.MaxNCS)  # Temporary array for gradation
        self.DLtemp = np.zeros(self.MaxNCS)  # Temporary array for distance
        self.PBEDIK = []  # Interpolated bed material gradation
        self.DM1FRT = np.zeros(self.MaxNGS)  # Group particle size (m)

        # Inlet and outlet flow conditions
        self.TimeTQ = []  # 各时段进口 T-Q 关系的T
        self.TimeQint = []  # 各时段进口 T-Q 关系的Q
        self.TimeTMP = []  # 各时段进口 T-Temp 关系的T
        self.TimeWTemp = []  # 各时段进口T-Temp 关系的Temp
        self.TimeTout = []   # 各时段出口 T-ZW 关系的T
        self.TimeZout = []   # 各时段出口 T-ZW 关系的Z
        self.TimeQout = []   # 各时段出口 Q-ZW 关系的Q
        self.ZRNOUT = []  # 出口断面水位流量关系 ZRNOUT(MaxZQ) dQ/dZ
        self.QRNOUT = []  # 出口断面水位流量关系 QRNOUT(MaxZQ) dQ/dZ
        self.DQZRN = []  # 出口断面水位流量关系 DQZRN dQ/dZ

        # Tributary flow conditions
        self.NcsTBY = []  # 支流所在河段的断面区间号码
        self.NumQtby = []  # 各支流T-Q过程的个数
        self.TimeTQtby = []  # 各支流的时间与流量过程中的时间
        self.TimeQtby = []  # 各支流的时间与流量过程中的流量

        # 区间引水沙量
        self.TimeWdiv = []  # 各河段的引水量(t,i)
        self.TimeSdiv = []  # 各河段的引沙量(t,i)
        self.TimeTdiv = []  # 引水时刻  (t)
        self.TimeTSdiv = []  # 引沙时刻  (t)  !zxlpr 新加入的
        self.QLold = []  # 侧向来流条件
        self.QLnew = []  # 侧向来流条件

        # Roughness for Control cross-section
        self.Control_CS_QD = [] #各控制断面段流量与糙率的关系曲线中的流量
        self.Control_CS_QDRN = [] #控制断面主槽糙率随流量的变化
        self.Control_CS_HighFP_Rg = [] #控制断面高低滩糙率
        self.Control_CS_LowFP_Rg = [] #控制断面高低滩糙率
        self.DHbed = []  # Roughness increment at control cross-sections 糙率增量
        self.DNbed = []  # Bed deformation thickness at control cross-sections 冲淤厚度

        # Roughness in Main channel
        self.CS_QDRN = []  # Interpolated roughness 4 main channel

        # Define arrays
        self.NumStby = np.zeros(self.MaxStp)
        self.TimeWDrch = np.zeros((self.MaxStp, self.MaxCTS))  # Flow rate at control cross-sections
        self.TimeSDrch = np.zeros((self.MaxStp, self.MaxCTS))  # Sediment load at control cross-sections
        self.TimeTT = np.zeros(self.MaxStp)  # Time array
        self.TimePP = np.zeros(self.MaxStp)  # Time array
        self.TimeSPK = np.zeros((self.MaxStp, self.MaxNGS))  # Suspended sediment gradation in main channel
        self.TimePKtby = np.zeros((self.MaxTBY, self.MaxStp, self.MaxNGS))  # Suspended sediment gradation in tributaries
        self.TSPMC = np.zeros(self.MaxStp)  # Time array for suspended sediment in main channel
        self.TSPKMC = np.zeros((self.MaxStp, self.MaxNGS))  # Suspended sediment gradation in main channel
        self.NSPtby = np.zeros(self.MaxTBY, dtype=int)  # Number of tributaries
        self.TSPtby = np.zeros((self.MaxTBY, self.MaxStp))  # Time array for suspended sediment in tributaries
        self.TSPKtby = np.zeros((self.MaxTBY, self.MaxStp, self.MaxNGS))  # Suspended sediment gradation in tributaries
        self.RNCTS = np.zeros((self.MaxCTS, self.MaxQN))  # Roughness of main channel at control cross-sections

        self.TempXX = np.zeros(self.MaxNCS)  # Temporary array for coordinates
        self.TempYY = np.zeros(self.MaxNCS)  # Temporary array for coordinates

        self.NTPML = np.zeros(self.MaxNCS)
        self.TimeTS = np.zeros(self.MaxStp)
        self.TimeSint = np.zeros(self.MaxStp)
        self.TimeSkint = np.zeros((self.MaxNGS, self.MaxStp))
        self.TimeTStby = np.zeros((self.MaxStp, self.MaxTBY))
        self.TimeStby = np.zeros((self.MaxStp, self.MaxTBY))
        self.TimeSKtby = np.zeros((self.MaxNGS, self.MaxStp, self.MaxTBY)) 
        self.TimeQSKtby = np.zeros((self.MaxNGS, self.MaxStp, self.MaxTBY))
        self.CSZW = np.zeros(self.MaxNCS)
        self.CSQQ = np.zeros(self.MaxNCS)
        self.CSUU = np.zeros(self.MaxNCS)
        self.CSHH = np.zeros(self.MaxNCS)
        self.CSAA = np.zeros(self.MaxNCS)
        self.CSBB = np.zeros(self.MaxNCS)
        self.CSZB = np.zeros(self.MaxNCS)
        self.CSSP = np.zeros(self.MaxNCS)
        self.CSRN = np.zeros(self.MaxNCS)
        self.CSUT = np.zeros(self.MaxNCS)
        self.CSWP = np.zeros(self.MaxNCS)
        self.CSRD = np.zeros(self.MaxNCS)
        self.CSFR = np.zeros(self.MaxNCS)
        self.CSQK = np.zeros(self.MaxNCS)
        self.CSQM = np.zeros(self.MaxNCS)
        self.CSUM = np.zeros(self.MaxNCS)
        self.CSHM = np.zeros(self.MaxNCS)
        self.CSAM = np.zeros(self.MaxNCS)
        self.CSBM = np.zeros(self.MaxNCS)
        self.CSQf = np.zeros(self.MaxNCS)
        self.CSUf = np.zeros(self.MaxNCS)
        self.CSHf = np.zeros(self.MaxNCS)
        self.CSAf = np.zeros(self.MaxNCS)
        self.CSBf = np.zeros(self.MaxNCS)

        self.CSSUS = np.zeros(self.MaxNCS)
        self.CSSCC = np.zeros(self.MaxNCS)
        self.SUSD50 = np.zeros(self.MaxNCS)
        self.SUSDPJ = np.zeros(self.MaxNCS)
        self.CSSVL = np.zeros(self.MaxNCS)
        self.CSSVM = np.zeros(self.MaxNCS)
        self.CSCNK = np.zeros(self.MaxNCS)
        self.WSmed = np.zeros(self.MaxNCS)

        self.DFCIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.TCnode = np.zeros((self.MaxNPT, self.MaxNCS))
        self.COKSD = np.zeros(self.MaxNCS)
        self.COMSD = np.zeros(self.MaxNCS)
        self.WSET = np.zeros(self.MaxNCS)
        self.WSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.UCIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.SKint = np.zeros(self.MaxStp)
        self.ALFSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.BLTSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.SUSIK1 = np.zeros(self.MaxNCS)
        self.SUSIK2 = np.zeros(self.MaxNCS)
        self.SUSIK3 = np.zeros(self.MaxNCS)
        self.SUSIK4 = np.zeros(self.MaxNCS)
        self.SUSIK5 = np.zeros(self.MaxNCS)
        self.SUSIK6 = np.zeros(self.MaxNCS)
        self.SUSIK7 = np.zeros(self.MaxNCS)
        self.SUSIK8 = np.zeros(self.MaxNCS)
        self.SUSIK9 = np.zeros(self.MaxNCS)
        self.SUSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.SCCIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.DPSCCIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.DPSUSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.QSLold = np.zeros((self.MaxNGS, self.MaxNCS))
        self.QSLnew = np.zeros((self.MaxNGS, self.MaxNCS))
        self.CSASIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.CSBSIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.BED50 = np.zeros(self.MaxNCS)
        self.BED90 = np.zeros(self.MaxNCS)
        self.BedPJ = np.zeros(self.MaxNCS)
        self.BED10 = np.zeros(self.MaxNCS)
        self.BED60 = np.zeros(self.MaxNCS)
        self.CSSL1 = np.zeros(self.MaxNCS)
        self.CSSL2 = [[0.0] * self.MaxTBY for _ in range(self.MaxNCS)]
        self.CSDA = np.zeros(self.MaxNCS)
        self.DAMC = np.zeros(self.MaxNCS)
        self.DAIJ = np.zeros((self.MaxNPT, self.MaxNCS))
        self.DZIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.DAIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.DHnode = np.zeros((self.MaxNPT, self.MaxNCS))
        self.DZnode = np.zeros((self.MaxNPT, self.MaxNCS))
        self.DZMC = np.zeros(self.MaxNCS)
        self.dASdt1 = np.zeros(self.MaxNCS)
        self.dASdt2 = np.zeros(self.MaxNCS)
        self.PDASIK1 = np.zeros((self.MaxNGS, self.MaxNCS))
        self.PDASIK2 = np.zeros((self.MaxNGS, self.MaxNCS))
        self.DASIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.VOLCSold = np.zeros(self.MaxNCS - 1)
        self.VOLMCold = np.zeros(self.MaxNCS - 1)
        self.TimeVLcs = np.zeros(self.MaxNCS)
        self.TimeVLMC = np.zeros(self.MaxNCS)
        self.TimeVL2CTCS = np.zeros((self.MaxCTS, self.MaxNCS)) 
        self.TimeVL2CTMC = np.zeros((self.MaxCTS, self.MaxNCS)) 
        self.CofNDX = np.zeros(self.MaxNCS)
        self.CofNDY = np.zeros(self.MaxNCS)
        self.DPBEDIK = np.zeros((self.MaxNGS, self.MaxNCS))
        self.THBAL = np.zeros(self.MaxNCS)
        self.THEML = []
        self.DPEML = []


        self.RNLFCS = np.zeros(self.MaxNCS)
        self.RNHFCS = np.zeros(self.MaxNCS)
        self.RNMC = np.zeros(self.MaxNCS)
        self.RNMCCS = np.zeros(self.MaxNCS)
        self.RNFPCS = np.zeros(self.MaxNCS)
        self.DNBDCS = np.zeros(self.MaxNCS)
        self.DHBDCS = np.zeros(self.MaxNCS)

        self.NoutCS = np.zeros(self.MaxNCS)
        self.ZTCSmax = np.zeros(self.MaxNCS)
        self.TZCSmax = np.zeros(self.MaxNCS)
        self.QTCSmax = np.zeros(self.MaxNCS)
        self.TQCSmax = np.zeros(self.MaxNCS)
        self.STCSmax = np.zeros(self.MaxNCS)
        self.TSCSmax = np.zeros(self.MaxNCS)
        self.CSHC = np.zeros(self.MaxNCS)
        self.CSpm = np.zeros(self.MaxNCS)
        self.Dsdt = np.zeros(self.MaxNCS)
        self.DAFDT = np.zeros(self.MaxNCS)
        self.NoutCS = np.zeros(self.MaxNCS)
        self.ZTCSmax = np.zeros(self.MaxNCS)
        self.TZCSmax = np.zeros(self.MaxNCS)
        self.QTCSmax = np.zeros(self.MaxNCS)
        self.TQCSmax = np.zeros(self.MaxNCS)
        self.STCSmax = np.zeros(self.MaxNCS)
        self.TSCSmax = np.zeros(self.MaxNCS)

    def import_para(self, Global_para_file: str, CS_profile: str, Flow_boundary_file: str, Roughness_file: str):

        # Input global parameters for hydrodynamics model
        print(f'----------------- Key step 1 -----------------\nRead the Global Parameter File')
        if not isinstance(Global_para_file, str):
            raise Exception('The global para file should be a filename under str type!')
        elif not Global_para_file.endswith('ALPRMT.dat'):
            raise Exception('The Global parameter file not ends with the ~ALPRMT.dat extension!')
        else:
            self._read_para_file(Global_para_file)
            self.ROI_name = 'Temp' if '_ALPRMT.dat' not in Global_para_file else os.path.basename(Global_para_file).split('_ALPRMT.dat')[0]
            try:
                self.input_path = os.path.join(self.work_env, self.ROI_name + '_Input\\')
                if Global_para_file != os.path.join(self.input_path, self.ROI_name + '_ALPRMT.dat':
                    shutil.copy(Global_para_file, os.path.join(self.input_path, self.ROI_name + '_ALPRMT.dat'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy global para file!')
        print(f'{str(self.ROI_name)}_AllPara.dat has been imported\n--------------- Key step 1 Done---------------')

        # Define the para based on the global parameter
        self.Coks = self.Cokm[self.MDSCC - 1][0]
        self.Coms = self.Cokm[self.MDSCC - 1][1]
        self.MaxStp = int(self.TimeSM * 3600.0 / self.DTstep)  # Total number of time steps

        # Input cross-section profile
        print(f'----------------- Key step 2 -----------------\nRead the Cross Section Profile')
        if not isinstance(CS_profile, str):
            raise Exception('The CS profile should be a filename under str type!')
        elif not CS_profile.endswith('CSProf.csv'):
            raise Exception('The cross section profile not ends with the CSProf.csv extension!')
        else:
            if (self.ROI_name is 'Temp' and '_CSProf.csv' not in CS_profile) or (self.ROI_name is not 'Temp' and f'{self.ROI_name}_CSProf.csv' in CS_profile):
                self._read_cs_file(CS_profile)
            else:
                raise Exception('The Cross profile does not share the same ROI name with Global parameter file! Double check!')
            try:
                if CS_profile != os.path.join(self.input_path, f'{self.ROI_name}_CSProf.csv'):
                    shutil.copy(CS_profile, os.path.join(self.input_path, f'{self.ROI_name}_CSProf.csv'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy cross_section profile!')
        print(f'{str(self.ROI_name)}_CSProf.Dat has been import\n--------------- Key step 2 Done---------------')

        # Input flow boundary
        print(f'----------------- Key step 3 -----------------\nRead the Flow Boundary Condition')
        if not isinstance(Flow_boundary_file, str):
            raise Exception('The Flow Boundary Condition should be a filename under str type!')
        elif not Flow_boundary_file.endswith('FlowBD.csv'):
            raise Exception('The Flow Boundary Condition not ends with the FlowBD.csv extension!')
        else:
            if (self.ROI_name is 'Temp' and '_FlowBD.csv' not in Flow_boundary_file) or (self.ROI_name is not 'Temp' and f'{self.ROI_name}_FlowBD.csv' in Flow_boundary_file):
                self._read_flow_file(Flow_boundary_file)
            else:
                raise Exception('The Flow boundary does not share the same ROI name with Global parameter file! Double check!')
            try:
                if Flow_boundary_file != os.path.join(self.input_path, f'{self.ROI_name}_FlowBD.csv'):
                    shutil.copy(Flow_boundary_file, os.path.join(self.input_path, f'{self.ROI_name}_FlowBD.csv'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy Flow Boundary Condition!')
        print(f'{str(self.ROI_name)}_FlowBD.csv has been import\n--------------- Key step 3 Done---------------')

        # Input the roughness file
        print(f'----------------- Key step 4 -----------------\nRead the {str(self.ROI_name)} roughness profile')
        if not isinstance(Roughness_file, str):
            raise Exception('The Roughness Profile should be a filename under str type!')
        elif not Roughness_file.endswith('Roughness.csv'):
            raise Exception('The Roughness Profile not ends with the Roughness.csv extension!')
        else:
            if (self.ROI_name is 'Temp' and '_Roughness.csv' not in Roughness_file) or (self.ROI_name is not 'Temp' and f'{self.ROI_name}_Roughness.csv' in Roughness_file):
                self._read_roughness_file(Roughness_file)
            else:
                raise Exception('The Roughness Profile does not share the same ROI name with Global parameter file! Double check!')
            try:
                if Roughness_file != os.path.join(self.input_path, f'{self.ROI_name}_Roughness.csv'):
                    shutil.copy(Roughness_file, os.path.join(self.input_path, f'{self.ROI_name}_Roughness.csv'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy Roughness Profile!')
        print(f'{str(self.ROI_name)}_Roughness.csv has been input\n--------------- Key step 4 Done---------------')

    def _read_para_file(self, para_file):
        with open(para_file, 'r') as f:
            file_inform = f.readlines()
            file_dic = {}
            for inform in file_inform:
                inform = inform.split('\n')[0].split(' #')[0]
                if ':' in inform:
                    file_dic[inform.split(':')[0]] = inform.split(':')[1]
                else:
                    print(str(inform))

        for _ in file_dic.keys():
            if _ in self.__dict__.keys():
                self.__dict__[_] = type(self.__dict__[_])(file_dic[_])
            else:
                raise ValueError(f'The input {str(_)} is not valid parameter!')

    def _read_cs_file(self, cs_file):

        # Read the cs file
        cs_num, cs_name_list, cs_control_list, cs_hydro, cs_id_all = 0, [], [], [], []
        pd_temp = pd.read_csv(cs_file, encoding='gbk', header=None)
        if len(pd_temp.keys()) > 4:
            pd_temp = pd_temp[pd_temp.keys()[0:4]]
            print('Only the first 4 columns were read')
        elif len(pd_temp.keys()) < 4:
            print('Not sufficient information in the cross section profile')
        pd_temp.columns = ['Id', 'Distance to left node', 'Ele', 'Type']

        # Get the profile of each cross-section
        _, cs_name_, maximum_node = 0, None, 0
        while _ < pd_temp.shape[0]:

            if str(pd_temp['Id'][_]).isnumeric():
                try:
                    cs_ = int(pd_temp['Id'][_])
                    cs_id_list.append(int(pd_temp['Id'][_]))
                    cs_dis2left_list.append(float(pd_temp['Distance to left node'][_]))
                    cs_ele_list.append(float(pd_temp['Ele'][_]))
                    cs_type_list.append(int(float(pd_temp['Type'][_])))
                except:
                    raise TypeError(f'The {str(cs_)} node of CSProf for {cs_name_} might be incorrect!')
            else:
                if pd_temp['Id'][_] != 'Id':
                    if cs_name_ is not None:
                        cs_id_all.append(cs_id_list)
                        self.XXIJ.append(cs_dis2left_list)
                        self.ZBIJ.append(cs_ele_list)
                        self.KNIJ.append(cs_type_list)
                        self.CS_node_num.append(len(cs_id_list))
                        maximum_node = max(maximum_node, len(cs_id_list))

                    cs_num += 1
                    cs_name_ = pd_temp['Id'][_]
                    cs_name_list.append(cs_name_)
                    try:
                        if len(self.CS_DistViaRiv) > 0 and self.CS_DistViaRiv[-1] > float(pd_temp['Distance to left node'][_]):
                            raise Exception(f'The sequence of the cross section {cs_name_} is not appropriate!')
                        self.CS_DistViaRiv.append(float(pd_temp['Distance to left node'][_]))
                    except:
                        raise Exception(f'Cross section Header for {cs_name_} is not appropriate!')

                    try:
                        cs_control_temp = bool(pd_temp['Ele'][_])
                        cs_control_list.append(cs_control_temp)
                    except:
                        raise Exception(f'The control indicator is not under the right type!')
                elif pd_temp['Id'][_] == 'Id':
                    cs_id_list, cs_dis2left_list, cs_ele_list, cs_type_list = [], [], [], []

            _ += 1

        # Check the consistency between different list
        if len(self.CS_DistViaRiv) != len(self.XXIJ) != len(self.ZBIJ) != len(self.KNIJ) != len(self.CS_node_num):
            raise Exception('The code error during input the cross section profile!')

        # Update the cross-section profile
        self.CS_name = cs_name_list
        self.Imax = cs_num
        self.MaxNCS = cs_num
        self.Jmax = maximum_node
        self.MaxNPT = maximum_node
        self.ZBINL = copy.deepcopy(self.ZBIJ)

        # Profile for control cross-sections
        self.Num_Control_CS = int(np.sum(np.array(cs_control_list).astype(np.int16)))
        self.MaxCTS = self.Num_Control_CS
        self.Control_CS_Id = [_ for _ in range(len(cs_control_list)) if cs_control_list[_]]
        self.Control_CS_DistViaRiv = [self.CS_DistViaRiv[_] for _ in self.Control_CS_Id]

        # Get the distance between the cross-section
        self.DX2CS = [self.CS_DistViaRiv[_ + 1] - self.CS_DistViaRiv[_] for _ in range(len(self.CS_DistViaRiv) - 1)]

        # Get the left right bank and channel number for each cross-section
        for _ in range(len(self.CS_node_num)):
            NMC_l, NMC_r, NodMCL, NodMCR = 0, 0, [], []
            for __ in range(self.CS_node_num[_] - 1):
                if (self.KNIJ[_][__] == 1 or self.KNIJ[_][__] == 2) and (self.KNIJ[_][__ + 1] == 0 or self.KNIJ[_][__ + 1] == 3):
                    NMC_l += 1
                    NodMCL.append(__)  # 主槽左侧滩地节点号
                elif (self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3) and (self.KNIJ[_][__ + 1] == 1 or self.KNIJ[_][__ + 1] == 2):
                    NMC_r += 1
                    NodMCR.append(__)  # 主槽右侧滩地节点号
            self.NodMCL.append(NodMCL)
            self.NodMCR.append(NodMCR)

            if NMC_l < NMC_r:
                raise Exception(f'The channel {self.cs_name[_]} has no left bank')
            elif NMC_l < NMC_r:
                raise Exception(f'The channel {self.cs_name[_]} has no right bank')
            elif NMC_l == 0:
                raise Exception(f'The left low floodplain of {self.cs_name[_]} might be missing')
            elif NMC_r == 0:
                raise Exception(f'The right low floodplain of {self.cs_name[_]} might be missing')
            elif NMC_l != NMC_r:
                raise Exception(f'The type of {self.cs_name[_]} is wrong')
            self.NMC1CS.append(NMC_l)

        # Check if the cross-section terrain is invalid
        # Generate the information of sub-reach
        for _ in range(self.Imax):
            DBIJ, DKIJ, SUMB = [], [], 0
            for __ in range(self.CS_node_num[_] - 1):
                DXJ = self.XXIJ[_][__ + 1] - self.XXIJ[_][__]
                if DXJ < 0:
                    raise Exception(f'The cross section {self.cs_name[_]} profile has invalid distance to left node!')
                elif DXJ < 0.01:
                    self.XXIJ[_][__ + 1] = self.XXIJ[_][__] + 0.01
                DBIJ.append(self.XXIJ[_][__ + 1] - self.XXIJ[_][__])
                DKIJ.append(self.KNIJ[_][__ + 1] * self.KNIJ[_][__])
                if self.KNIJ[_][__ + 1] * self.KNIJ[_][__] != 2 and self.KNIJ[_][__ + 1] * self.KNIJ[_][__] != 4:
                    SUMB += self.XXIJ[_][__ + 1] - self.XXIJ[_][__]
            self.DBIJ.append(DBIJ)
            self.DKIJ.append(DKIJ)
            self.BWMC.append(SUMB)

        # Compute the bankfull elevation
        zlbk = np.zeros(self.Imax)
        zrbk = np.zeros(self.Imax)
        for _ in range(self.Imax):  # 左侧滩地高程
            for __ in range(self.CS_node_num[_] - 1):
                if (self.KNIJ[_][__] + self.KNIJ[_][__ + 1] == 1) or (self.KNIJ[_][__] + self.KNIJ[_][__ + 1] == 4):  # (1,0) Or(1,3)
                    zbtc = max(self.ZBIJ[_][__], self.ZBIJ[_][__ + 1])  # 滩槽平均高程
                    zlbk[_] = zbtc
                    break

        for _ in range(self.Imax):  # 右侧滩地高程
            for __ in range(self.CS_node_num[_] - 1, 0, -1):
                if (self.KNIJ[_][__] + self.KNIJ[_][__ - 1] == 1) or (self.KNIJ[_][__] + self.KNIJ[_][__ - 1] == 4):  # (0,1) Or(3,1)
                    zbtc = max(self.ZBIJ[_][__], self.ZBIJ[_][__ - 1]) # 滩槽平均高程
                    zrbk[_] = zbtc
                    break

        for i in range(self.Imax):
            self.ZBBF.append(min(zlbk[i], zrbk[i]))

        # Compute the lowest elevation for each cross-section
        for _ in range(self.Imax):
            ZB, = 10000.0
            for __ in range(self.CS_node_num[_]):
                if (self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3) and self.ZBIJ[_][__] < ZB:
                    ZB = self.ZBIJ[_][__]
            self.CSZBMN.append(ZB)  # For Main Channel

        for _ in range(self.Imax):
            SUMA, SUMB = 0.0, 0.0
            for __ in range(self.CS_node_num[_] - 1):
                if self.KNIJ[_][__] * self.KNIJ[_][__ + 1] != 2 and self.KNIJ[_][__] * self.KNIJ[_][__ + 1] != 4:
                    DZ1 = self.ZBIJ[_][__] - self.CSZBMN[_]
                    DZ2 = self.ZBIJ[_][__ + 1] - self.CSZBMN[_]
                    SUMA += 0.5 * (DZ1 + DZ2) * self.DBIJ[_][__]
                    SUMB += self.DBIJ[_][__]
            self.CSZBav.append(self.CSZBMN[_] + SUMA / SUMB)
            self.CSZBav0.append(copy.deepcopy(self.CSZBav[_]))  # Initial record

        # Output section starting distance and elevation
        node_list, dis2left_list, elevation_list, type_list = [], [], [], []
        for _ in range(self.Imax):
            node_list.append(f'CS name: {cs_name_list[_]}')
            dis2left_list.append(f'Distance: {self.CS_DistViaRiv[_]} km')
            elevation_list.append(f'Main channel width: {self.BWMC[_]} m')
            type_list.append(f'Mean channel ele: {self.CSZBav[_]} m')

            for __ in range(self.CS_node_num[_]):
                node_list.append([__ + 1])
                dis2left_list.append(self.XXIJ[_][__])
                elevation_list.append(self.ZBIJ[_][__])
                type_list.append(self.KNIJ[_][__])
        output_section_ele = pd.DataFrame({'Column1': node_list, 'Column2': dis2left_list, 'Column3': elevation_list, 'Column4': type_list})
        output_section_ele.to_csv(os.path.join(self.output_path, f'{self.ROI_name}_INICSProf.csv'), index=False, header=False)

        # Output section main channel area starting distance and elevation zxl
        cs_list, DIST_list, BMC_list, ZBk_list, ZBav_list = [], [], [], [], []
        for I in range(0, self.Imax):
            cs_list.append(I)
            DIST_list.append(self.CS_DistViaRiv[_])
            BMC_list.append(self.BWMC[_])
            ZBk_list.append(self.ZBBF[_])
            ZBav_list.append(self.CSZBav[_])

        output_section_ele = pd.DataFrame({'I': cs_list, 'DIST(km)': DIST_list, 'BMC(m)': BMC_list, 'ZBk(m)': ZBk_list, 'ZBav(m)': ZBav_list})
        output_section_ele.to_csv(os.path.join(self.output_path, f'{self.ROI_name}_INICSZBMC.csv'), index=False, header=True)

    def _read_flow_file(self, flowfile):

        pd_temp = pd.read_csv(flowfile, encoding='gbk', header=None)
        if len(pd_temp.keys()) > 4:
            pd_temp = pd_temp[pd_temp.keys()[0:4]]
            print('Only the first 4 columns were read')
        elif len(pd_temp.keys()) < 4:
            print('Not sufficient information in the cross section profile')
        pd_temp.columns = ['Time_step', 'Hour', 'Q&Z', 'Tributary']

        # Get the profile of each cross-section
        _, tri_cs_list = 0, []
        tri_all_id, tri_all_time, tri_all_qt = [], [], []
        nfl_all_id, nfl_all_time, nfl_all_qt = [], [], []
        inlet_factor, inlet_temp_factor, outlet_factor = False, False, False
        while _ < pd_temp.shape[0]:

            # 入口流量-时间关系
            if pd_temp['Time_step'][_] == 'inlet-Q':

                if inlet_factor:
                    raise Exception('Two inlet flow conditions were imported twice!')

                inlet_id, inlet_time, inlet_q = [], [], []
                if pd_temp['Hour'][_] not in self.cs_name:
                    raise Exception(f"The profile of inlet cross-section {pd_temp['Hour'][_]} is not imported")
                elif pd_temp['Hour'][_] != self.cs_name[0]:
                    raise Exception(f"The profile of inlet cross-section {pd_temp['Hour'][_]} is not consistent with the first section {self.cs_name[0]} of dem")

                if pd_temp['Q&Z'][_] == 'h':
                    simu_factor = 3600
                elif pd_temp['Q&Z'][_] == 'min':
                    simu_factor = 60
                elif pd_temp['Q&Z'][_] == 's':
                    simu_factor = 1
                else:
                    raise Exception(f"The simulation time scale {str(pd_temp['Q&Z'][_])} is not supported!")

                __ = copy.deepcopy(_) + 2
                try:
                    t = int(pd_temp['Time_step'][__])
                except:
                    raise TypeError('The inlet-Q should under the standard format!')

                while __ < pd_temp.shape[0]:
                    try:
                        inlet_id.append(int(pd_temp['Time_step'][__]))
                        inlet_time.append(float(pd_temp['Hour'][__]) * simu_factor)
                        inlet_q.append(float(pd_temp['Q&Z'][__]))
                        __ += 1
                    except:
                        self.TimeTQ = inlet_time
                        self.TimeQint = inlet_q
                        self.NumTQint = len(inlet_id)
                        inlet_factor = True
                        break

            # 入口水温-时间关系
            elif pd_temp['Time_step'][_] == 'inlet-T':

                if inlet_temp_factor:
                    raise Exception('Two inlet temperature flow conditions were imported twice!')

                inlet_TEM_id, inlet_TEM_time, inlet_TEM = [], [], []
                if pd_temp['Hour'][_] not in self.cs_name:
                    raise Exception(f"The profile of inlet-T cross-section {pd_temp['Hour'][_]} is not imported")
                elif pd_temp['Hour'][_] != self.cs_name[0]:
                    raise Exception(f"The profile of inlet-T cross-section {pd_temp['Hour'][_]} is not consistent with the first section {self.cs_name[0]} of dem")

                if pd_temp['Q&Z'][_] == 'h':
                    simu_factor = 3600
                elif pd_temp['Q&Z'][_] == 'min':
                    simu_factor = 60
                elif pd_temp['Q&Z'][_] == 's':
                    simu_factor = 1
                else:
                    raise Exception(f"The simulation time scale {str(pd_temp['Q&Z'][_])} is not supported!")

                __ = copy.deepcopy(_) + 2
                try:
                    t = int(pd_temp['Time_step'][__])
                except:
                    raise TypeError('The inlet-T should under the standard format!')

                while __ < pd_temp.shape[0]:
                    try:
                        inlet_TEM_id.append(int(pd_temp['Time_step'][__]))
                        inlet_TEM_time.append(float(pd_temp['Hour'][__])  * simu_factor)
                        inlet_TEM.append(float(pd_temp['Q&Z'][__]))
                        __ += 1
                    except:
                        self.TimeTMP = inlet_TEM_time
                        self.TimeWTemp = inlet_TEM
                        self.NumTMP = len(inlet_TEM_id)
                        inlet_temp_factor = True
                        break

            # 出口水位-流量-时间关系
            elif pd_temp['Time_step'][_] == 'outlet-Z':

                if outlet_factor:
                    raise Exception('Two outlet flow conditions were imported twice!')

                if pd_temp['Hour'][_] not in self.cs_name:
                    raise Exception(f"The profile of outlet-Z cross-section {pd_temp['Hour'][_]} is not imported")
                elif pd_temp['Hour'][_] != self.cs_name[-1]:
                    raise Exception( f"The profile of outlet-Z cross-section {pd_temp['Hour'][_]} is not consistent with the last cross section {self.cs_name[0]} of dem")

                if pd_temp['Q&Z'][_] == 'h':
                    simu_factor = 3600
                elif pd_temp['Q&Z'][_] == 'min':
                    simu_factor = 60
                elif pd_temp['Q&Z'][_] == 's':
                    simu_factor = 1
                else:
                    raise Exception(f"The simulation time scale {str(pd_temp['Q&Z'][_])} is not supported!")

                if pd_temp['Tributary'][_] in ['T-Q', 'T-Z', 'Q-Z']:
                    self.KBDout, outlet_id, outlet_time, outlet_Q, outlet_Z = pd_temp['Tributary'][_], [], [], [], []
                else:
                    raise Exception(f"The input relationship {str(pd_temp['Tributary'][_])} is not supported!")

                __ = copy.deepcopy(_) + 2
                try:
                    t = int(pd_temp['Time_step'][__])
                except:
                    raise TypeError('The inlet-T should under the standard format!')

                while __ < pd_temp.shape[0]:
                    try:
                        outlet_id.append(int(pd_temp['Time_step'][__]))
                        if self.KBDout == 'T-Q':
                            outlet_time.append(float(pd_temp['Hour'][__]) * simu_factor)
                            outlet_Q.append(float(pd_temp['Q&Z'][__]))
                        elif self.KBDout == 'T-Z':
                            outlet_time.append(float(pd_temp['Hour'][__]) * simu_factor)
                            outlet_Z.append(float(pd_temp['Q&Z'][__]))
                        elif self.KBDout == 'Q-Z':
                            outlet_Q.append(float(pd_temp['Hour'][__]))
                            outlet_Z.append(float(pd_temp['Q&Z'][__]))
                        __ += 1
                    except:
                        if self.KBDout == 'T-Q':
                            self.TimeTout = outlet_time
                            self.TimeQout = outlet_Q
                        elif self.KBDout == 'T-Z':
                            self.TimeTout = outlet_time
                            self.TimeZout = outlet_Z
                        elif self.KBDout == 'Q-Z':
                            self.ZRNOUT = outlet_Z
                            self.QRNOUT = outlet_Q

                        self.NumTQout = len(outlet_time)
                        self.NumTZout = len(outlet_time)
                        self.NumZQout = len(outlet_Z)
                        outlet_factor = True

                        if self.KBDout == 'Q-Z':
                            self.DQZRN = []
                            for _ in range(self.NumZQout):
                                if _ == 0:
                                    DZ12 = self.ZRNout[_ + 1] - self.ZRNout[_]
                                    if DZ12 <= 1.0E-3:
                                        print(f'k,DZ12={str(_)}, {str(DZ12)} in Sub_inputdata')
                                    self.DQZRN.append((self.QRNout[_ + 1] - self.QRNout[_]) / (self.ZRNout[_ + 1] - self.ZRNout[_]))
                                elif 0 < _ < self.NumZQout - 1:
                                    DZ12 = self.ZRNout[_ + 1] - self.ZRNout[_ - 1]
                                    if DZ12 <= 1.0E-3:
                                        print(f'k,DZ12={str(_)}, {str(DZ12)} in Sub_inputdata')
                                    self.DQZRN.append((self.QRNout[_ + 1] - self.QRNout[_ - 1]) / (self.ZRNout[_ + 1] - self.ZRNout[_ - 1]))
                                elif _ == self.NumZQout - 1:
                                    DZ12 = self.ZRNout[_] - self.ZRNout[_ - 1]
                                    if DZ12 <= 1.0E-3:
                                        print(f'k,DZ12={str(_)}, {str(DZ12)} in Sub_inputdata')
                                    self.DQZRN.append((self.QRNout[_] - self.QRNout[_ - 1]) / (self.ZRNout[_] - self.ZRNout[_ - 1]))
                        break

            # 支流流量-时间情况
            elif pd_temp['Time_step'][_] == 'Tribu-Z':

                if pd_temp['Hour'][_] in self.cs_name:
                    tri_cs_name = pd_temp['Time_step'][_]
                    tri_cs_list.append(self.cs_name.index(tri_cs_name))
                else:
                    raise Exception(f"The profile of Tribu-Z cross-section {pd_temp['Hour'][_]} is not imported")

                if pd_temp['Q&Z'][_] == 'h':
                    simu_factor = 3600
                elif pd_temp['Q&Z'][_] == 'min':
                    simu_factor = 60
                elif pd_temp['Q&Z'][_] == 's':
                    simu_factor = 1
                else:
                    raise Exception(f"The simulation time scale {str(pd_temp['Q&Z'][_])} is not supported!")

                tri_id, tri_T, tri_qt = [], [], []
                __ = copy.deepcopy(_) + 2
                try:
                    t = int(pd_temp['Time_step'][__])
                except:
                    raise TypeError('The Tribu-Z should under the standard format!')

                while __ < pd_temp.shape[0]:
                    try:
                        tri_id.append(int(pd_temp['Time_step'][__]))
                        tri_T.append(float(pd_temp['Hour'][__])  * simu_factor)
                        tri_qt.append(float(pd_temp['Q&Z'][__]))
                        __ += 1
                    except:
                        tri_all_id.append(tri_id)
                        tri_all_time.append(tri_all_time)
                        tri_all_qt.append(tri_qt)
                        break

            # 区间引水-时间信息
            elif pd_temp['Time_step'][_] == 'NFL-Q':
                raise Exception('The NFL part has not been transformed yet.')

            _ += 1

        # Update the flow parameter
        self.Num_TBY = len(tri_all_id)
        self.NcsTBY = tri_cs_list
        self.NFLDiv = False # 引水引沙参数=0(不考虑),=1(考虑)
        self.TimeTQtby = tri_all_time
        self.TimeQtby = tri_all_qt
        self.NumQtby = [len(_) for _ in tri_all_id]
        self.QLnew = [0.0 for _ in range(self.Imax)]
        self.QLold = [0.0 for _ in range(self.Imax)]

        if 'T' in self.KBDout and self.TimeTout != self.TimeQint:
            raise ValueError('The time for inlet and outlet flow conditions were not consistent!')
        if self.NcsTBY > 0 and False in [_ == self.TimeTQtby[0] for _ in self.TimeTQtby]:
            raise ValueError('The tributary flow conditions were not consistent!')
        if self.NcsTBY > 0 and self.TimeTQtby[0] != self.TimeQint:
            raise ValueError('The tributary flow conditions and inlet time were not consistent!')

    def _read_roughness_file(self, roughness_file):

        # Read the roughness file
        try:
            pd_temp = pd.read_csv(roughness_file, header=None)
        except:
            print(traceback.format_exc())
            raise Exception('Some error occurred during reading the roughness file!')

        if len(pd_temp.keys()) < self.Num_Control_CS + 1:
            raise Exception('Not sufficient information in the roughness profile')

        # Generate the columns name
        columns_t = ['Factor']
        for _ in range(len(pd_temp.keys()) - 1):
            columns_t.append(f'cs_{str(_)}')
        pd_temp.columns = columns_t

        _ = 0
        self.Control_CS_QDRN = [[] for _ in range(len(self.Control_CS_Name))]
        self.Control_CS_LowFP_Rg = [None for _ in range(len(self.Control_CS_Name))]
        self.Control_CS_HighFP_Rg = [None for _ in range(len(self.Control_CS_Name))]
        while _ < pd_temp.shape[0]:
            if str(pd_temp['Factor'][_]) == 'Qcon-R':

                # Check the consistency between control cs name and roughness cs
                control_cs_list = list(pd_temp[_]).remove('Qcon-R')
                consis_list = [__ in control_cs_list for __ in self.Control_CS_Name]

                if False in consis_list:
                    raise Exception(f'The Roughness for control cross section {self.Control_CS_Name[consis_list.index(False)]} is missing!')
                while True:
                    _ += 1
                    if not str(pd_temp['Factor'][_]).isnumeric() or _ >= pd_temp.shape[0]:
                        break
                    else:
                        self.Control_CS_QD.append(float(pd_temp['Factor'][_]))
                        for __ in self.Control_CS_Name:
                            if str(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]).isnumeric():
                                self.Control_CS_QDRN[self.Control_CS_Name.index(__)].append(float(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]))
                            else:
                                raise Exception('Non number shown in control cs roughness file')

            elif str(pd_temp['Factor'][_]) == 'Type-R':
                FP1_factor, FP2_factor = False, False
                control_cs_list = list(pd_temp[_]).remove('Type-R')
                consis_list = [__ in control_cs_list for __ in self.Control_CS_Name]

                if False in consis_list:
                    raise Exception(f'The roughness of high and low floodplains for control cross section {self.Control_CS_Name[consis_list.index(False)]} are missing!')
                while True:
                    _ += 1
                    if str(pd_temp['Factor'][_]) == 'FP1':
                        for __ in self.Control_CS_Name:
                            if str(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]).isnumeric():
                                self.Control_CS_HighFP_Rg[self.Control_CS_Name.index(__)] = float(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_])
                            else:
                                raise Exception('Non number shown in control cs roughness file')
                        FP1_factor = True

                    elif str(pd_temp['Factor'][_]) == 'FP2':
                        for __ in self.Control_CS_Name:
                            if str(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]).isnumeric():
                                self.Control_CS_LowFP_Rg[self.Control_CS_Name.index(__)] = float(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_])
                            else:
                                raise Exception('Non number shown in control cs roughness file')
                        FP2_factor = True

                    else:
                        if FP1_factor and FP2_factor:
                            break
                        else:
                            raise Exception('The roughness for high or low floodplain were missed!')

            elif str(pd_temp['Factor'][_]) == 'DRN-DH':
                DRN_factor, DH_factor = False, False
                control_cs_list = list(pd_temp[_]).remove('DRN-DH')
                consis_list = [__ in control_cs_list for __ in self.Control_CS_Name]

                if False in consis_list:
                    raise Exception(f'The roughness of high and low floodplains for control cross section {self.Control_CS_Name[consis_list.index(False)]} are missing!')
                while True:
                    _ += 1
                    if str(pd_temp['Factor'][_]) == 'DRN':
                        for __ in self.Control_CS_Name:
                            if str(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]).isnumeric():
                                self.DNbed[self.Control_CS_Name.index(__)] = float(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_])
                            else:
                                raise Exception('Non number shown in control cs roughness file')
                        DRN_factor = True

                    elif str(pd_temp['Factor'][_]) == 'DH(m)':
                        for __ in self.Control_CS_Name:
                            if str(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_]).isnumeric():
                                self.DHbed[self.Control_CS_Name.index(__)] = float(pd_temp[f'cs_{str(control_cs_list.index(__))}'][_])
                            else:
                                raise Exception('Non number shown in control cs roughness file')
                        DH_factor = True

                    else:
                        if DRN_factor and DH_factor:
                            break
                        else:
                            raise Exception('The DRN or DH were missed!')

        # Roughness in Main Channel
        for _ in range(len(self.CS_DistViaRiv)):
            for _ in range(len(self.Control_CS_QD)):
            self.
        for k in range(num_qn):
            temp_yy = [rncts[n][k] for n in range(nctcs)]
            for i in range(imax):
                rnmncs[i, k] = np.interp(dist_lg[i], temp_xx, temp_yy)

        # Roughness on low floodplain
        rnlfcs = np.interp(dist_lg[:imax], temp_xx, rnlwfp)

        # Roughness on high floodplain
        rnhfcs = np.interp(dist_lg[:imax], temp_xx, rnhgfp)
        #
        # # Roughness increment
        # dnbdcs = np.interp(dist_lg[:imax], temp_xx, dnbed)        self.RNMNCS = []
        #
        # # Erosion/deposition increment
        # dhbdcs = np.interp(dist_lg[:imax], temp_xx, dhbed)
        #
        # print("7---File=*_QNReLT.Dat has been input")
        #
        # if num_qn >= max_qn:
        #     print(f"Maximum number of Q-N relationships: {max_qn}")
        #     print(f"Actual number of Q-N relationships: {num_qn}")
        #     input("Press Enter to continue...")
        #
        # # Read output cross-section numbers
        # fullnam = filnam[:namlen] + '_NoutCS.Dat'
        # with open(fullnam, 'r') as f:
        #     next(f)  # Skip first line
        #     num_ocs = int(next(f))
        #     if num_ocs > 0:
        #         next(f)  # Skip line
        #         nout_cs = [int(next(f).split()[1]) for _ in range(num_ocs)]
        #
        # # Initialize maximum values for cross-sections
        # ztcs_max = np.full(imax, -10000.0)
        # tzcs_max = np.zeros(imax)
        # qtcs_max = np.full(imax, -10000.0)
        # tqcs_max = np.zeros(imax)
        # stcs_max = np.full(imax, -10000.0)
        # tscs_max = np.zeros(imax)

    def calculate(self):

        # Initialize parameters
        Istep = 0
        NumRem = 0  # Number of times to calculate water and sediment conditions and topography changes
        TimeSC = 0.0  # Total calculation time (seconds)
        TimeHR = 0.0  # Total calculation time (hours)

        # Calculate initial riverbed elevation and output results
        def OutputCSProfiles(NumRem):
            pass

        OutputCSProfiles(NumRem)
        print("1----------OutputCSProfiles")

        # Record bed sediment gradation and output results
        def OutputBEDMTGRAD(NumRem):
            pass

        OutputBEDMTGRAD(NumRem)
        print("3----------OutputBEDMTGRAD")

        # Set-up for the spin-up period
        if NYspin == 1:
            NFLBDtp = NFLBD
            NFLGAtp = NFLGA
            SPtime = 0.0  # Initial simulation time

        # Input initial cross-section flow and water level
        def Initialdata():
            pass

        Initialdata()
        print("4----------Initialdata")

        # Calculate sediment transport capacity under initial water and sediment conditions
        def CompVISCOS(TimeWtemp, ViSTFW):
            pass

        def CompSETVEL():
            pass

        def COMPCHBEDSZ():
            pass

        def Comp1DSEDCC():
            pass

        def COMPSUSnd():
            pass

        CompVISCOS(TimeWtemp[0], ViSTFW)
        CompSETVEL()
        COMPCHBEDSZ()
        Comp1DSEDCC()
        COMPSUSnd()

        if NYSpin == 0:
            def WriteCSsedi():
                pass

            WriteCSsedi()

        # Main calculation loop for water and sediment changes and riverbed deformation
        for Istep in range(1, NStep + 1):
            # Code for the spin-up period
            if NYspin == 1:
                SPtime += DTStep / 3600.0

            if NYspin == 1 and SPtime <= TimeSP:
                NFLBD = 0
                NFLGA = 0
                Pleft = 100.0 - 100 * SPtime / Timesp
                print(f"Spin-up Period, SPtime,Left={SPtime},{Pleft}%")

            if NYspin == 1 and SPtime > TimeSP:
                NFLBD = NFLBDtp
                NFLGA = NFLGAtp

            NFLRD = 0
            TimeSC = DTStep * Istep  # Cumulative calculation time (seconds)
            TimeHR = TimeSC / 3600.0  # Cumulative calculation time (hours)

            # Determine whether to record calculation results
            if Istep % NtRD == 0:
                NumRem += 1
                NFLRD = 1

            # Determine boundary conditions for 1D flow calculation
            if Nshow == 1:
                print("1 COMP1DFLOWBD")

            def COMP1DFLOWBD():
                pass

            COMP1DFLOWBD()

            # Calculate 1D steady water surface profile
            if Nshow == 1:
                print("2 COMP1DFLOWRT")

            def SOL1DFLOWRT():
                pass

            SOL1DFLOWRT()

            # Record flow conditions if needed
            if NFLRD == 1:
                def OUTPUT1DFLOW(NumRem):
                    pass

                OUTPUT1DFLOW(NumRem)

            if int(TimeSC) % (3600 * 24) == 0:
                def WriteCSflow():
                    pass

                WriteCSflow()

            # Just After the spin-up period
            if NYspin == 1:
                if SPtime <= TimeSP and (SPtime + DTStep / 3600.0) > TimeSP:
                    OUTPUT1DFLOW(0)
                    WriteCSflow()

            # Sediment transport calculation
            if Nshow == 1:
                print("Into Sub-SedBD")

            if NFLST == 1:
                def COMP1DSEDBD():
                    pass

                def CompVISCOS(Wtemp, ViSTFW):
                    pass

                def CompSETVEL():
                    pass

                COMP1DSEDBD()
                CompVISCOS(Wtemp, ViSTFW)
                CompSETVEL()

            if Nshow == 1:
                print("3  SOL1DSEDTPEQ ")

            if NFLST == 1:
                def SOL1DSEDTPEQ():
                    pass

                def COMPSUSnd():
                    pass

                SOL1DSEDTPEQ()
                COMPSUSnd()

            if Nshow == 1:
                print("31  Comp1DSEDCC/CompDASDT(2)  ")

            if NFLST == 1:
                def Comp1DSEDCC():
                    pass

                def CompDASDT(n):
                    pass

                Comp1DSEDCC()
                CompDASDT(2)

            if Nshow == 1:
                print("32 COMPSUSnd/SCND ")

            if NFLST == 1 and NFLRD == 1:
                def Output1DSed(NumRem):
                    pass

                Output1DSed(NumRem)

            if NFLST == 1:
                if int(TimeSC) % (3600 * 24) == 0:
                    WriteCSsedi()

            # Just After the spin-up period
            if NYspin == 1:
                if SPtime <= TimeSP and (SPtime + DTStep / 3600.0) > TimeSP:
                    Output1DSed(0)
                    WriteCSsedi()

            if NFLST == 1 and NFLBD == 1 and MDLDA == 4:
                COMPSUSnd()

        # Note: Many functions are defined as pass statements as their implementations are not provided in the original code.
        # You may need to implement these functions based on your specific requirements.

    def SOL1DFLOWRT(self, Hmin, DTStep, CitaFw, QinT, QLnew, Zout, Qout, DQdZout, Npoint, XXIJ, ZBIJ,
                    DistLG, DX2CS, KNIJ, RNIJ, Zbmin, Qold, Zold, Aold, Bold, DMKold, ALFold):

        Qnew = np.zeros(self.Imax)
        Znew = np.zeros(self.Imax)

        for i in range(self.Imax):
            Zold[i] = self.CSZW[i]
            Qold[i] = self.CSQQ[i]

        # CompQZBA1(Hmin, DHmin, CitaFw, Imax, Jmax, NPT1CS, KNIJ, RNIJ, DBIJ, ZBIJ, Zbmin, DX2CS, Zold, Aold, Bold,
        #           DMKold, ALFold)
        #
        # PreissmannScheme(Qold, Zold, Bold, Aold, DMKold, ALFold, Qnew, Znew)
        #
        # CompFlowPRMTs(Znew, Qnew)
        #
        # for i in range(Imax):
        #     DAFDT[i] = (CSAA[i] - Aold[i]) / DTstep
        #
        # return Qnew, Znew


if __name__ == "__main__":
    model = Morphodynamic_Model1D()
    model.input_data('G:\\A_Landsat_Floodplain_veg\\Hydrodynamic_model\\')
