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


def compute_cof(Bup: float, Bdown: float):
    """
    Calculate contraction or expansion coefficient
    :param Bup: Upstream water surface width
    :param Bdown: Downstream water surface width
    :return: Coefficient value
    """
    # Constants
    DB_MIN = 1.5
    DB_MAX = 4.0
    COF_1 = 3.0  # Expansion coefficient
    COF_2 = 1.0  # Contraction coefficient

    if Bdown > Bup:  # Expansion
        db = Bdown / Bup
        if db <= DB_MIN:
            cof = 0.0
        elif db <= DB_MAX:
            cof = COF_1 * (db - DB_MIN) / (DB_MAX - DB_MIN)
        else:
            cof = COF_1

    else: #Contraction
        db = Bup / Bdown
        if db <= DB_MIN:
            cof = 0.0
        elif db <= DB_MAX:
            cof = COF_2 * (db - DB_MIN) / (DB_MAX - DB_MIN)
        else:
            cof = COF_2

    return cof


def cubic_spline_interpolation(X, Y, T):

    K = -1
    Z = 0.0
    if len(X) == 0:
        raise Exception('The cubic spline interpolation need 3 set of data at least')
    elif len(X) == 1:
        return Y[0]
    elif len(X) == 2:
        return (Y[0] * (T - X[1]) - Y[1] * (T - X[0])) / (X[0] - X[1])
    else:
        if T <= X[1]:
            K = 1
        elif T >= X[-2]:
            K = len(X) - 1
        else:
            K = 1
            M = len(X)
            while abs(K - M) != 1:
                L = (K + M) // 2
                if T < X[L]:
                    M = L
                else:
                    K = L
        if K >= len(X):
            K = len(X) - 1

        U3 = (Y[K + 1] - Y[K]) / (X[K + 1] - X[K])
        if len(X) == 3:
            if K == 1:
                U4 = (Y[2] - Y[1]) / (X[2] - X[1])
                U5 = 2.0 * U4 - U3
                U2 = 2.0 * U3 - U4
                U1 = 2.0 * U2 - U3
            elif K == 2:
                U2 = (Y[1] - Y[0]) / (X[1] - X[0])
                U1 = 2.0 * U2 - U3
                U4 = 2.0 * U3 - U2
                U5 = 2.0 * U4 - U3
        else:
            if K <= 2:
                U4 = (Y[K + 2] - Y[K + 1]) / (X[K + 2] - X[K + 1])
                if K == 2:
                    U2 = (Y[1] - Y[0]) / (X[1] - X[0])
                    U1 = 2 * U2 - U3
                    if len(X) == 4:
                        U5 = 2.0 * U4 - U3
                    else:
                        U5 = (Y[4] - Y[3]) / (X[4] - X[3])
                else:
                    U2 = 2 * U3 - U4
                    U1 = 2 * U2 - U3
                    U5 = (Y[3] - Y[2]) / (X[3] - X[2])
            elif K >= (len(X) - 2):
                U2 = (Y[K] - Y[K - 1]) / (X[K] - X[K - 1])
                if K == (len(X)- 2):
                    U4 = (Y[len(X) - 1] - Y[len(X) - 2]) / (X[len(X) - 1] - X[len(X) - 2])
                    U5 = 2 * U4 - U3
                    if len(X) == 4:
                        U1 = 2.0 * U2 - U3
                    else:
                        U1 = (Y[K - 1] - Y[K - 2]) / (X[K - 1] - X[K - 2])
                else:
                    U4 = 2 * U3 - U2
                    U5 = 2 * U4 - U3
                U1 = (Y[K - 1] - Y[K - 2]) / (X[K - 1] - X[K - 2])
            else:
                U2 = (Y[K] - Y[K - 1]) / (X[K] - X[K - 1])
                U1 = (Y[K - 1] - Y[K - 2]) / (X[K - 1] - X[K - 2])
                U4 = (Y[K + 2] - Y[K + 1]) / (X[K + 2] - X[K + 1])
                U5 = (Y[K + 3] - Y[K + 2]) / (X[K + 3] - X[K + 2])

    A = abs(U4 - U3)
    B = abs(U1 - U2)

    if (A + 1.0 == 1.0) and (B + 1.0 == 1.0):
        G1 = (U2 + U3) / 2.0
    else:
        G1 = (A * U2 + B * U3) / (A + B)

    A = abs(U4 - U5)
    B = abs(U3 - U2)
    if (A + 1.0 == 1.0) and (B + 1.0 == 1.0):
        G2 = (U3 + U4) / 2.0
    else:
        G2 = (A * U3 + B * U4) / (A + B)

    A = Y[K]
    B = G1
    D = X[K + 1] - X[K]
    C = (3 * U3 - 2 * G1 - G2) / D
    D = (G2 + G1 - 2 * U3) / (D * D)
    S = T - X[K]
    Z = A + B * S + C * S * S + D * S * S * S

    return Z


def linear_interpolation(X: list, Y: list, T: float):

    Z = None
    if T < X[0]:
        Z = Y[0]
    elif T > X[-1]:
        Z = Y[-1]
    elif X[0] <= T <= X[-1]:
        for i in range(len(X) - 1):
            if (T - X[i]) * (T - X[i + 1]) <= 0:
                if (X[i] - X[i + 1]) == 0.0:
                    Z = Y[i]
                else:
                    DK = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
                    Z = Y[i] + DK * (T - X[i])
                break
    else:
        raise Exception('Code Err during the linear interpolation')

    if Z is None:
        raise Exception('Code Err during the linear interpolation')
    else:
        return Z


def compute_water_depth(ZBJ, KNJ, ZW, Hmin):
    # Get the Water depth at each node
    HHJ = [ZW - ZBJ[_] if ZW > ZBJ[_] + Hmin else 0 for _ in range(len(ZBJ))]
    for _ in range(len(HHJ)):
        if HHJ[_] != 0 and (KNJ[_] == 1 or KNJ[_] == 2):
            left_indi, right_indi = False, False
            for __ in range(1, len(HHJ)):
                if _ - __ > 0:
                    if HHJ[_ - __] == 0:
                        break
                    elif HHJ[_ - __] > 0 and KNJ[_ - __] == 0:
                        left_indi = True
                    else:
                        pass
                else:
                    break

            for __ in range(1, len(HHJ)):
                if _ + __ < len(HHJ) + 1:
                    if HHJ[_ + __] == 0:
                        break
                    elif HHJ[_ + __] > 0 and KNJ[_ + __] == 0:
                        right_indi = True
                    else:
                        pass
                else:
                    break

            if left_indi or right_indi:
                pass
            else:
                HHJ[_] = 0
    return HHJ


def compute_channel_char(Hmin, DHmin, DBJ, ZBJ, KNJ, RNJ, ZW, simplified_factor = False):
    """

    :param Hmin:
    :param DHmin:
    :param DBJ: Distance between two nodes(i.e, Nij Nij+1) of this cross-section
    :param ZBJ: Elevation for each node of this cross-section
    :param KNJ: Channel or floodplain indicator for each node of this cross-section
    :param RNJ: Roughness for each node of this cross-section
    :param ZW: Water level at this cross-section
    :return:
    """

    # Define arrs
    kNJ = [0 for _ in range(len(DBJ))]
    HHJ = compute_water_depth(ZBJ, KNJ, ZW, Hmin)

    # Define the distance, area, wetness area, hydraulic diameter
    BB = 0.0
    AA = 0.0
    DMK = 0.0
    ALF1 = 0.0

    for _ in range(len(RNJ) - 1):
        Manning_coef = (RNJ[_] + RNJ[_ + 1]) / 2.0 # Manning_coef

        if HHJ[_] > Hmin and HHJ[_ + 1] > Hmin:
            DB = DBJ[_]  # 子断面的宽度
            DA = DB * (HHJ[_] + HHJ[_ + 1]) * 1/2  # 子断面的面积
            WP = np.sqrt(DB ** 2 + (HHJ[_] - HHJ[_ + 1]) ** 2)  # 子断面的湿周
            RD = DA / WP  # 子断面的水力半径

        elif HHJ[_] <= Hmin and HHJ[_ + 1] <= Hmin:
            DB = 0.0
            DA = 0.0
            WP = 0.0
            RD = 0.0

        elif HHJ[_] <= Hmin and HHJ[_ + 1] > Hmin:
            DB = HHJ[_ + 1] / (ZBJ[_] - ZBJ[_ + 1]) * DBJ[_]
            DA = DB * HHJ[_ + 1] * 0.5
            WP = np.sqrt(DB ** 2 + HHJ[_ + 1] ** 2)
            RD = DA / WP

        elif HHJ[_] > Hmin and HHJ[_ + 1] <= Hmin:
            DB = HHJ[_] / (ZBJ[_ + 1] - ZBJ[_]) * DBJ[_]
            DA = DB * HHJ[_] * 0.5
            WP = np.sqrt(DB ** 2 + HHJ[_] ** 2)
            RD = DA / WP

        else:
            raise Exception('Code err during the computation area')

        if RD > 0.0:
            if (KNJ[_] == 0 or KNJ[_] == 3) and DA / RD <= DHmin:
                Manning_coef = 0.020
            DK = (DA / Manning_coef) * (RD ** (2/3)) # 子段面流量模数
        else:
            DK = 0.0

        if DA > 0.0:
            ALF1 += (DK ** 3) / (DA ** 2)
        else:
            ALF1 += 0.0

        AA += DA
        BB += DB
        DMK += DK

    if AA > 0.0:
        ALF2 = (DMK**3.0) / (AA**2.0)
        Alf = ALF1 / ALF2
    else:
        Alf = 1.0

    if simplified_factor:
        return AA
    else:
        return AA, BB, DMK, Alf


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
        self.para_output_path = None

        # User-defined para
        self.ROI_name: str = ''

        # Define maximum number
        self.MaxNCS = 100  # Maximum number of cross-sections
        self.MaxNGS = 10  # Maximum number of sediment groups 最大泥沙分组数
        self.MaxStp = 1000  # Maximum number of time steps
        self.MaxCTS = 100  # Maximum number of control cross-sections
        self.MaxTBY = 10  # Maximum number of tributaries
        self.MaxQN = 10  # Maximum number of flow rates for roughness calculation 最大数量 (流量与糙率的关系曲线)
        self.MaxNPT = 0  # 最大节点数
        self.MaxNMC = 0  # 断面内的最大主槽划分数 (MaxNMC<=3)
        self.MaxZQ = 0  # 最大数量 (出口水位流量关系曲线)

        # Define the global parameters
        self.Imax: int = 0  # 计算断面数
        self.Jmax: int = 0  # 断面上最大节点数
        self.TimeSM: float = 0.0  # 模拟时间(hour)
        self.NYspin: bool = True
        self.TimeSP: float = 0.0  # 设置好的Spin-up time (hours)
        self.SPTime: float = 0.0  # 计算中的Spin-up time (hours)
        self.DTstep: int = 30  # 计算固定的时间步长(Unit: Second) 45 不可算
        self.NTRD: int = 480  # 记录计算结果的时间步数
        self.NumTBY: int = 0  # 实际支流数量
        self.NumCCS: int = 0  # 控制断面数量
        self.NumGS: int = 12  # 非均匀沙分组数
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

        # Parameters during the calculation
        self.Istep = 0  # 第ISTEP时间步
        self.NumRem = 0  # Number of times to calculate water and sediment conditions and topography changes
        self.TimeSC = 0.0  # 已计算时间(Second)
        self.TimeHR = 0.0  # 已计算时间(Hour)

        #
        self.NumQN = 0  # 实际QN关系的个数
        self.NumTSint = 0  # 主流进口断面T-S个数
        self.NumTdiv = 0  # 沿程Time-Q过程的个数
        self.NumTsdiv = 0  # 沿程Time-S过程的个数 !ZXLPR新加入的 原来只有NumTdiv, !沿程Time-Q/S过程的个数
        self.NumOCS = 0  # 要输出结果的断面数量
        self.Wtemp = 0.0
        self.VistFW = 0.0

        # Define the sediment parameters
        self.SedTrans_flag: bool = True  # 泥沙输移是否计算(True = 不计算 / False = 计算)  Flag for Sediment Transport
        self.BedDeform_flag: bool = True  # 河床冲淤是否计算(True = 不计算 / False = 计算)  Flag for bed deformation
        self.GradAdj_flag: bool = True  # 计算过程中床沙级配是否调整(True = 不计算 / False = 计算) Flag for Gradation Adjustment
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
        self.CS_num = 0
        self.CS_name = []  # Cross-section name
        self.CS_DistViaRiv = []  # Distance along the river for each cross-section
        self.CS_node_num = []  # Node number for each cross-section
        self.ZBINL = []  # Elevation for each node on each cross-section (Initial value)
        self.NMC1CS = []  # Number of the main channel for each cross-section
        self.NodMCL = []  # Node of left bank for each channel of each cross-section主槽左侧滩地节点号
        self.NodMCR = []  # Node of right bank for each channel of each cross-section 主槽右侧滩地节点号
        self.BWMC = []  # Width of main channel for each cross-section
        self.ZBBF = []  # Bankfull elevation for each cross-section
        self.DX2CS = []  # Distance between two cross-sections
        self.CSZBmn = []  # The lowest elevation (Thalweg) for each cross-section
        self.CSZBav = []  # The average elevation of the main channel
        self.CSZBav0 = []  # Initial record of the average elevation for the main channel

        # Cross-section hydraulic para calculated based on CS profile and flow data
        self.CSZW = []  # Water level for each cross-section
        self.CSQQ = []  # Discharge for each cross-section
        self.CSAA = []  # Flow area for each cross-section 过水面积
        self.CSBB = []  # Flow width for each cross-section 水面宽度
        self.CSQK = []  # 流量模数 for each cross-section
        self.CSWP = []  # wetness area for each cross-section 湿周
        self.CSRD = []  # 水力半径 for each cross-section
        self.CSHH = []  # Water depth for each cross-section
        self.CSHC = []  # 形心下水深 for each cross-section
        self.CSZB = []  # 平均河底高程(水面以下) for each cross-section
        self.CSUU = []  # Flow velocity for each cross-section
        self.CSSP = []  # 能坡 for each cross-section
        self.CSRN = []  # Average roughness for each cross-section
        self.CSUT = []  # 摩阻流速 for the channel of each cross-section
        self.CSFR = []  # Froude number of each cross-section

        # Main-channel hydraulic para calculated based on CS profile and flow data
        self.CSQM = []  # Discharge for the main channel of each cross-section
        self.CSAM = []  # Flow area for the main channel of each cross-section
        self.CSBM = []  # Width for the main channel of each cross-section
        self.CSUM = []  # Flow velocity for the main channel of each cross-section
        self.CSHM = []  # Water depth for the main channel of each cross-section

        # Floodplain hydraulic para calculated based on CS profile and flow data
        self.CSQF = []  # Discharge for the FLOODPLAIN of each cross-section
        self.CSAF = []  # Flow area for the FLOODPLAIN of each cross-section
        self.CSBF = []  # Width for the FLOODPLAIN of each cross-section
        self.CSUF = []  # Flow velocity for the FLOODPLAIN of each cross-section
        self.CSHF = []  # Water depth for the FLOODPLAIN of each cross-section

        # Node-level inform
        self.XXIJ = []  # Distance to the left bank for each node on each cross-section
        self.ZBIJ = []  # Elevation for each node on each cross-section
        self.KNIJ = []  # Channel or floodplain indicator for each node on each cross-section
        self.RNIJ = []  # Roughness at node level
        self.DBIJ = []  # Distance between two nodes(i.e, Nij Nij+1) on each cross-section
        self.DKIJ = []  # 各子断面代号(0*1)(0*2)均为主槽
        self.AAIJ = []  # 各子断面的过水面积
        self.QKIJ = []  # 各子断面的流量模数
        self.QQIJ = []  # 各子断面的流量
        self.UUIJ = []  # 各子断面的流速
        self.BBIJ = []  # 各子断面过水的水面宽度
        self.HHij = []  # 各子断面过水的水面水深
        self.UTIJ = []
        self.WPIJ = []  # 各子断面的湿周
        self.RDIJ = []  # 各子断面的水力半径
        self.Hnode = []  # 各节点水深
        self.Unode = []  # 各节点流速
        self.Qnode = []  # 各节点单宽流量
        self.UTnode = []  # 各节点摩阻流速
        self.TFnode = []  # 各节点水流切应力

        # Control cross-section profile
        self.Control_CS_Id = []  # ID for each control cross-section
        self.Control_CS_Name = [] # Name for each control cross-section
        self.Control_CS_DistViaRiv = []  # Distance along river for each control cross-section
        
        # Intermediate para
        self.DQDZoutlet_new = 0.0
        self.Qinlet_new = 0.0
        self.Zoutlet_new = 0.0
        self.Qoutlet_new = 0.0
        self.QL_new = []  # 侧向来流条件
        
        # Bed Material Gradation
        self.DMSiev = np.zeros(self.MaxNGS)  # 筛分粒径
        self.NoCSsp = np.zeros(self.MaxNCS, dtype=int)  # Cross-sections with bed gradation
        self.DistSP = np.zeros(self.MaxNCS)  # Distance of cross-sections with bed gradation
        self.PBEDSP = np.zeros((self.MaxNCS, self.MaxNGS))  # Bed gradation at cross-sections
        self.PKtemp = np.zeros(self.MaxNCS)  # Temporary array for gradation
        self.DLtemp = np.zeros(self.MaxNCS)  # Temporary array for distance
        self.PBEDIK = []  # Interpolated bed material gradation
        self.DM1FRT = np.zeros(self.MaxNGS)  # Group particle size (m)

        # Inlet and outlet flow boundary conditions
        self.KBDout = None  # 下游边界类型
        self.NumTQint = 0  # 进口Time-Qint过程的个数
        self.NumTMP = 0  # 进口Time-Wtemp过程的个数
        self.NumTQout = 0  # 出口Time-Qout过程的个数
        self.NumTZout = 0  # 出口Time-Zout过程的个数
        self.NumZQout = 0  # 出口水位流量关系的个数
        self.TimeTQ = []  # 各时段进口 T-Q 关系的T
        self.TimeQint = []  # 各时段进口 T-Q 关系的Q
        self.TimeTMP = []  # 各时段进口 T-Temp 关系的T
        self.TimeWTemp = []  # 各时段进口T-Temp 关系的Temp
        self.TimeTout = []   # 各时段出口 T-ZW 关系的T
        self.TimeZout = []   # 各时段出口 T-ZW 关系的Z
        self.TimeQout = []   # 各时段出口 Q-ZW 关系的Q
        self.ZRNout = []  # 出口断面水位流量关系 ZRNOUT(MaxZQ) dQ/dZ
        self.QRNout = []  # 出口断面水位流量关系 QRNOUT(MaxZQ) dQ/dZ
        self.DQZRN = []  # 出口断面水位流量关系 DQZRN dQ/dZ

        # Tributary flow boundary conditions
        self.NcsTBY = []  # 支流所在河段的断面区间号码
        self.NumQtby = []  # 各支流T-Q过程的个数
        self.TimeTtby = []  # 各支流的时间与流量过程中的时间
        self.TimeQtby = []  # 各支流的时间与流量过程中的流量

        # 区间引水沙量
        self.TimeWdiv = []  # 各河段的引水量(t,i)
        self.TimeSdiv = []  # 各河段的引沙量(t,i)
        self.TimeTdiv = []  # 引水时刻  (t)
        self.TimeTSdiv = []  # 引沙时刻  (t)  !zxlpr 新加入的
        self.QLold = []  # 侧向来流条件

        # Roughness and dh/dn for Control cross-section
        self.Control_CS_QD = []  # 各控制断面段流量与糙率的关系曲线中的流量
        self.Control_CS_QDRN = []  # 控制断面主槽糙率随流量的变化
        self.Control_CS_Hfloodplain_RN = []  # 控制断面高低滩糙率
        self.Control_CS_Lfloodplain_RN = []  # 控制断面高低滩糙率
        self.Control_CS_DHbed = []  # Roughness increment at control cross-sections 糙率增量
        self.Control_CS_DNbed = []  # Bed deformation thickness at control cross-sections 冲淤厚度
        self.Control_CS_Init_Q = [] # Initial discharge (m3/s) at control cross-sections 冲淤厚度
        self.Control_CS_Init_Z = [] # Initial water level (m) at control cross-sections 冲淤厚度

        # Roughness in Main channel
        self.CS_QDRN = []  # Interpolated roughness 4 main channel
        self.CS_Hfloodplain_RN = []  # Interpolated high floodplain roughness 4 main channel
        self.CS_Lfloodplain_RN = []  # Interpolated low floodplain roughness 4 main channel
        self.CS_DHbed = []  # Roughness increment 4 main channel 糙率增量
        self.CS_DNbed = []  # Bed deformation thickness 4 main channel 冲淤厚度

        # Bed deformation para
        self.DZMC = [] # Bed deformation for main channel of each cross-section
        self.DASDT1 = []  #=dA0/dt
        self.DNBDCS = [] # 各断面的糙率增量
        self.DHBDCS = []  # 各断面的冲淤幅度增量

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

        self.NoutCS = np.zeros(self.MaxNCS)
        self.ZTCSmax = np.zeros(self.MaxNCS)
        self.TZCSmax = np.zeros(self.MaxNCS)
        self.QTCSmax = np.zeros(self.MaxNCS)
        self.TQCSmax = np.zeros(self.MaxNCS)
        self.STCSmax = np.zeros(self.MaxNCS)
        self.TSCSmax = np.zeros(self.MaxNCS)

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

    def import_para(self, Global_para_file: str, CS_profile: str,  Flow_boundary_file: str, Roughness_file: str, Bed_material_file: str=None):

        # Input global parameters for hydrodynamics model
        print(f'----------------- Key step 1 -----------------\nRead the Global Parameter File')
        if not isinstance(Global_para_file, str):
            raise Exception('The global para file should be a filename under str type!')
        elif not Global_para_file.endswith('_GLBpara.dat'):
            raise Exception('The Global parameter file not ends with the ~GLBpara.dat extension!')
        else:
            self._read_para_file(Global_para_file)
            if self.ROI_name is None:
                print('Please define the ROI_name in global para file!')
            self.ROI_name.replace(' ', '')
            try:
                self.input_path = os.path.join(self.work_env, self.ROI_name + '_Input\\')
                if Global_para_file != os.path.join(self.input_path, self.ROI_name + '_GLBpara.dat'):
                    shutil.copy(Global_para_file, os.path.join(self.input_path, self.ROI_name + '_GLBpara.dat'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy global para file!')
        print(f'{str(self.ROI_name)}_GLBpara.dat has been imported\n--------------- Key step 1 Done---------------')

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
            self._read_cs_file(CS_profile)
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
        elif not Flow_boundary_file.endswith('_FlwBound.csv'):
            raise Exception('The Flow Boundary Condition not ends with the _FlowINPUT.csv extension!')
        else:
            self._read_flow_input(Flow_boundary_file)
            try:
                if Flow_boundary_file != os.path.join(self.input_path, f'{self.ROI_name}_FlwBound.csv'):
                    shutil.copy(Flow_boundary_file, os.path.join(self.input_path, f'{self.ROI_name}_FlwBound.csv'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy Flow Boundary Condition!')

        # 初始值设定侧向入流条件 Initiate the side inflow
        self.QL_new = [0.0 for _ in range(self.Imax-1)]
        self.QLold = [0.0 for _ in range(self.Imax-1)]
        print(f'{str(self.ROI_name)}_FlwBound.csv has been import\n--------------- Key step 3 Done---------------')

        # Input the roughness file
        print(f'----------------- Key step 4 -----------------\nRead the {str(self.ROI_name)} roughness profile')
        if not isinstance(Roughness_file, str):
            raise Exception('The Roughness Profile should be a filename under str type!')
        elif not Roughness_file.endswith('_QNRelt.csv'):
            raise Exception('The Roughness Profile not ends with the _QNRelt.csv extension!')
        else:
            self._read_roughness_file(Roughness_file)
            try:
                if Roughness_file != os.path.join(self.input_path, f'{self.ROI_name}_QNRelt.csv'):
                    shutil.copy(Roughness_file, os.path.join(self.input_path, f'{self.ROI_name}_QNRelt.csv'))
            except:
                print(traceback.format_exc())
                raise Exception('Error during copy Roughness Profile!')
        print(f'{str(self.ROI_name)}_QNRelt.csv has been input\n--------------- Key step 4 Done---------------')

        # Input bed material
        if Bed_material_file is not None:
            print(f'----------------- Optional step 1 -----------------\nRead the Bed material')
            if not isinstance(Bed_material_file, str):
                raise Exception('The Bed_material_file should be a filename under str type!')
            elif not Bed_material_file.endswith('_BMGrad.csv'):
                raise Exception('The Bed_material_file not ends with the BMGrad.csv extension!')
            else:
                self._read_flow_file(Bed_material_file)
                try:
                    if Bed_material_file != os.path.join(self.input_path, f'{self.ROI_name}_BMGrad.csv'):
                        shutil.copy(Bed_material_file, os.path.join(self.input_path, f'{self.ROI_name}_BMGrad.csv'))
                except:
                    print(traceback.format_exc())
                    raise Exception('Error during copy BMGrad Condition!')
            print(f'{str(self.ROI_name)}_QNRelt.csv has been import\n--------------- Optional step 1 Done---------------')

        # Input sediment
        if Bed_material_file is not None:
            print(f'----------------- Optional step 2 -----------------\nRead the sediment')
            if not isinstance(Bed_material_file, str):
                raise Exception('The Bed_material_file should be a filename under str type!')
            elif not Bed_material_file.endswith('_BMGrad.csv'):
                raise Exception('The Bed_material_file not ends with the BMGrad.csv extension!')
            else:
                self._read_flow_file(Bed_material_file)
                try:
                    if Bed_material_file != os.path.join(self.input_path, f'{self.ROI_name}_BMGrad.csv'):
                        shutil.copy(Bed_material_file, os.path.join(self.input_path, f'{self.ROI_name}_BMGrad.csv'))
                except:
                    print(traceback.format_exc())
                    raise Exception('Error during copy BMGrad Condition!')
            print(f'{str(self.ROI_name)}_QNRelt.csv has been import\n--------------- Optional step 2 Done---------------')

    def write_para(self):
        pass
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

        # Read the cs file and Update the cross-section profile
        cs_name_list, cs_control_list, cs_hydro, cs_id_all, cs_control_station = [], [], [], [], []
        try:
            pd_temp = pd.read_csv(cs_file, header=None, encoding='gbk')
        except UnicodeError:
            pd_temp = pd.read_csv(cs_file, header=None)
        except:
            raise Exception('The cs profile cannot be read')

        if len(pd_temp.keys()) > 4:
            pd_temp = pd_temp[pd_temp.keys()[0:4]]
            print('Only the first 4 columns were read')
        elif len(pd_temp.keys()) < 4:
            print('Not sufficient information in the cross section profile')
        pd_temp.columns = ['Id', 'Distance to left node', 'Ele', 'Type']

        # Get the profile of each cross-section
        maximum_node = 0
        cs_start_index = list(pd_temp[pd_temp['Id'] == 'Id'].index)

        # Check the cs profile
        if len(cs_start_index) == 0:
            raise Exception('No valid cs profile is imported')
        else:
            self.Imax = len(cs_start_index)
            self.MaxNCS = len(cs_start_index)

        # Derive the profile
        cs_id_list = []
        for _ in cs_start_index:
            if not 0 <= _ - 1 <= pd_temp.shape[0]:
                raise Exception('The CSProfile ID is not in the right format!')
            else:
                try:
                    cs_name_list.append(str(pd_temp['Id'][_ - 1]))
                    if len(self.CS_DistViaRiv) > 0 and self.CS_DistViaRiv[-1] > float(pd_temp['Distance to left node'][_ - 1]):
                        raise Exception(f'The sequence of the cross section {cs_name_list[-1]} is not appropriate!')
                    self.CS_DistViaRiv.append(float(pd_temp['Distance to left node'][_ - 1]))
                    cs_control_list.append(pd_temp['Ele'][_ - 1].upper() == 'TRUE')
                    cs_control_station.append(pd_temp['Type'][_ - 1])
                except:
                    raise TypeError(f'The header of CSProf for cross section {str(_)} might be incorrect!')

                try:
                    if cs_start_index.index(_) == len(cs_start_index) - 1:
                        end_index = pd_temp.shape[0]
                    else:
                        end_index = cs_start_index[cs_start_index.index(_) + 1] - 1

                    cs_id_list.append([int(__) for __ in pd_temp['Id'][_ + 1: end_index]])
                    self.XXIJ.append([float(__) for __ in pd_temp['Distance to left node'][_ + 1: end_index]])
                    self.ZBIJ.append([float(__) for __ in pd_temp['Ele'][_ + 1: end_index]])
                    self.KNIJ.append([int(float(__)) for __ in pd_temp['Type'][_ + 1: end_index]])
                    self.CS_node_num.append(len(cs_id_list[-1]))
                    maximum_node = max(maximum_node, len(cs_id_list[-1]))
                except:
                    raise TypeError(f'The file of CSProf for cross section {str(_)} might be incorrect!')

        # Check the consistency between different list
        if len(self.CS_DistViaRiv) != len(self.XXIJ) != len(self.ZBIJ) != len(self.KNIJ) != len(self.CS_node_num):
            raise Exception('The code error during input the cross section profile!')

        # Update the cross-section profile
        self.CS_num = len(cs_name_list)
        self.CS_name = cs_name_list
        self.Jmax = maximum_node
        self.MaxNPT = maximum_node
        self.ZBINL = copy.deepcopy(self.ZBIJ)

        # Profile for control cross-sections
        self.NumCCS = int(np.sum(np.array(cs_control_list).astype(np.int16)))
        self.MaxCTS = self.NumCCS
        self.Control_CS_Name = [self.CS_name[_] for _ in range(len(cs_control_list)) if cs_control_list[_]]
        self.Control_CS_Id = [_ for _ in range(len(cs_control_list)) if cs_control_list[_]]
        self.Control_CS_DistViaRiv = [self.CS_DistViaRiv[_] for _ in self.Control_CS_Id]
        self.Control_CS_station = [cs_control_station[_] for _ in range(len(cs_control_list)) if cs_control_list[_]]

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
            self.NodMCL.extend(NodMCL)
            self.NodMCR.extend(NodMCR)

            if NMC_l < NMC_r:
                raise Exception(f'The channel {self.CS_name[_]} has no left bank')
            elif NMC_l < NMC_r:
                raise Exception(f'The channel {self.CS_name[_]} has no right bank')
            elif NMC_l == 0:
                raise Exception(f'The left low floodplain of {self.CS_name[_]} might be missing')
            elif NMC_r == 0:
                raise Exception(f'The right low floodplain of {self.CS_name[_]} might be missing')
            elif NMC_l != NMC_r:
                raise Exception(f'The type of {self.CS_name[_]} is wrong')
            self.NMC1CS.append(NMC_l)

        # Check if the cross-section terrain is invalid
        # Generate the information of sub-reach
        for _ in range(self.Imax):
            DBIJ, DKIJ, SUMB, RNIJ = [], [], 0, []  # 各子断面宽度, 各子断面代号(0*1)(0*2)均为主槽, Width of main channel for each cross-section
            for __ in range(self.CS_node_num[_] - 1):

                DXJ = self.XXIJ[_][__ + 1] - self.XXIJ[_][__]
                if DXJ < 0:
                    raise Exception(f'The cross section {self.CS_name[_]} profile has invalid distance to left node!')
                elif DXJ < 0.01:
                    self.XXIJ[_][__ + 1] = self.XXIJ[_][__] + 0.01
                DBIJ.append(self.XXIJ[_][__ + 1] - self.XXIJ[_][__])
                DKIJ.append(self.KNIJ[_][__ + 1] * self.KNIJ[_][__])

                if self.KNIJ[_][__ + 1] * self.KNIJ[_][__] != 2 and self.KNIJ[_][__ + 1] * self.KNIJ[_][__] != 4:
                    SUMB += self.XXIJ[_][__ + 1] - self.XXIJ[_][__]

                if self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3:
                    RNIJ.append(0.020)
                elif self.KNIJ[_][__] == 1:
                    RNIJ.append(0.030)
                elif self.KNIJ[_][__] == 2:
                    RNIJ.append(0.040)

            self.DBIJ.append(DBIJ)
            self.DKIJ.append(DKIJ)
            self.BWMC.append(SUMB)
            self.RNIJ.append(RNIJ)

        # Compute the bank-full elevation
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
            self.ZBBF.append(min(zlbk[i], zrbk[i]))  # bank full elevation

        # Compute the lowest elevation for each cross-section
        for _ in range(self.Imax):
            ZB = 10000.0
            for __ in range(self.CS_node_num[_]):
                if (self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3) and self.ZBIJ[_][__] < ZB:
                    ZB = self.ZBIJ[_][__]
            self.CSZBmn.append(ZB)  # For Main Channel # Thalweg elevation

        # Compute the average elevation of the channel
        for _ in range(self.Imax):
            SUMA, SUMB = 0.0, 0.0
            for __ in range(self.CS_node_num[_] - 1):
                if self.KNIJ[_][__] * self.KNIJ[_][__ + 1] != 2 and self.KNIJ[_][__] * self.KNIJ[_][__ + 1] != 4:
                    DZ1 = self.ZBIJ[_][__] - self.CSZBmn[_]
                    DZ2 = self.ZBIJ[_][__ + 1] - self.CSZBmn[_]
                    SUMA += 0.5 * (DZ1 + DZ2) * self.DBIJ[_][__]
                    SUMB += self.DBIJ[_][__]
            self.CSZBav.append(self.CSZBmn[_] + SUMA / SUMB)
        self.CSZBav0 = copy.deepcopy(self.CSZBav) # Initial record

    def _read_flow_input(self, flowfile):

        # Read the flow boundary condition
        try:
            pd_temp = pd.read_csv(flowfile, header=None, encoding='gbk')
        except UnicodeError:
            pd_temp = pd.read_csv(flowfile, header=None)
        except:
            raise Exception('The flow boundary condition cannot be read')

        if len(pd_temp.keys()) > 4:
            pd_temp = pd_temp[pd_temp.keys()[0:4]]
            print('Only the first 4 columns were read')
        elif len(pd_temp.keys()) < 4:
            print('Not sufficient information in the cross section profile')
        pd_temp.columns = ['Time_step', 'Hour', 'Q&Z', 'Tributary']

        # Get the flow profile of inlet
        tri_cs_list = []
        tri_all_id, tri_all_time, tri_all_q = [], [], []
        nfl_all_id, nfl_all_time, nfl_all_qt = [], [], []
        inlet_factor, inlet_temp_factor, outlet_factor, distance_inflow_factor, station_init_factor = False, False, False, False, False
        id_index = list(pd_temp[pd_temp['Time_step'] == 'k'].index)
        for _ in id_index:

            # Import the inlet discharge - time
            if pd_temp['Time_step'][_ - 1] == 'inlet_Q-T':

                # Check if inlet cs is valid
                if inlet_factor:
                    raise Exception('Two inlet flow conditions were imported twice!')

                if pd_temp['Hour'][_ - 1] not in self.CS_name:
                    raise Exception(f"The profile of inlet cross-section {pd_temp['Hour'][_ - 1]} is not imported")
                elif pd_temp['Hour'][_ - 1] != self.CS_name[0]:
                    raise Exception(f"The profile of inlet cross-section {pd_temp['Hour'][_ - 1]} is not consistent with the first section {self.CS_name[0]} of dem")

                # Derive the Q-T
                start_index = _ + 1
                end_index = pd_temp.shape[0] if id_index.index(_) == len(id_index) - 1 else id_index[id_index.index(_) + 1] - 1
                try:
                    inlet_id = [int(pd_temp['Time_step'][__]) for __ in range(start_index, end_index)]
                    self.TimeTQ = [float(pd_temp['Hour'][__]) * 3600 for __ in range(start_index, end_index)]
                    self.TimeQint = [float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)]
                    self.NumTQint = len(inlet_id)
                    inlet_factor = True
                except:
                    raise TypeError('The inlet_Q-T should under the standard format!')

            # Import the inlet water temperature - time
            elif pd_temp['Time_step'][_ - 1] == 'inlet-T-T':

                if inlet_temp_factor:
                    raise Exception('Two inlet temperature flow conditions were imported twice!')

                if pd_temp['Hour'][_ - 1] not in self.CS_name:
                    raise Exception(f"The profile of inlet-T {pd_temp['Hour'][_ - 1]} is not imported")
                elif pd_temp['Hour'][_ - 1] != self.CS_name[0]:
                    raise Exception(f"The profile of inlet-T {pd_temp['Hour'][_ - 1]} is not consistent with the first section {self.CS_name[0]} of dem")

                # Derive the T-T
                start_index = _ + 1
                end_index = pd_temp.shape[0] if id_index.index(_) == len(id_index) - 1 else id_index[id_index.index(_) + 1] - 1

                try:
                    inlet_TEM_id = [int(pd_temp['Time_step'][__]) for __ in range(start_index, end_index)]
                    self.TimeTMP = [float(pd_temp['Hour'][__]) * 3600 for __ in range(start_index, end_index)]
                    self.TimeWTemp = [float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)]
                    self.NumTMP = len(inlet_TEM_id)
                    inlet_temp_factor = True
                except:
                    raise TypeError('The inlet_T-T should under the standard format!')

            # Import the outlet water lvl - discharge - time
            elif pd_temp['Time_step'][_ - 1].startswith('outlet_'):

                # Check if the outlet data is valid
                if outlet_factor:
                    raise Exception('Two outlet flow conditions were imported twice!')

                if pd_temp['Hour'][_ - 1] not in self.CS_name:
                    raise Exception(f"The profile of outlet cross-section {pd_temp['Hour'][_ - 1]} is not imported")
                elif pd_temp['Hour'][_ - 1] != self.CS_name[-1]:
                    raise Exception(f"The profile of outlet cross-section {pd_temp['Hour'][_ - 1]} is not consistent with the first section {self.CS_name[-1]} of dem")

                outlet_type = pd_temp['Time_step'][_ - 1].split('_')[1]
                if outlet_type in ['Q-T', 'Z-T', 'Q-Z']:
                    self.KBDout, outlet_id, outlet_time, outlet_Q, outlet_Z = outlet_type, [], [], [], []
                else:
                    raise Exception(f"The input relationship {str(outlet_type)} is not supported!")

                # Derive the outlet Q-Z-T
                start_index = _ + 1
                end_index = pd_temp.shape[0] if id_index.index(_) == len(id_index) - 1 else id_index[id_index.index(_) + 1] - 1

                try:
                    time_step = [int(pd_temp['Time_step'][__]) for __ in range(start_index, end_index)]
                    self.TimeTout = [float(pd_temp['Hour'][__]) * 3600 for __ in range(start_index, end_index)]

                    if self.KBDout == 'Q-T':
                        self.TimeQout = [float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)]
                        self.NumTQout = len(self.TimeQout)

                    elif self.KBDout == 'Z-T':
                        self.TimeZout = [float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)]
                        self.NumTZout = len(self.TimeZout)

                    elif self.KBDout == 'Q-Z':
                        self.QRNout = [float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)]
                        self.ZRNout = [float(pd_temp['Tributary'][__]) for __ in range(start_index, end_index)]
                        self.NumZQout = len(self.QRNout)

                        # Generate the dQ/dZ Relation
                        self.DQZRN = []
                        for _ in range(self.NumZQout):
                            if _ == 0:
                                DZ12 = self.ZRNout[_ + 1] - self.ZRNout[_]
                                if DZ12 <= 1.0E-3:
                                    raise ValueError(f'Given the interval in water level is too small dQ/dz relationship is not valid')
                                self.DQZRN.append((self.QRNout[_ + 1] - self.QRNout[_]) / (self.ZRNout[_ + 1] - self.ZRNout[_]))
                            elif 0 < _ < self.NumZQout - 1:
                                DZ12 = self.ZRNout[_ + 1] - self.ZRNout[_ - 1]
                                if DZ12 <= 1.0E-3:
                                    raise ValueError(f'Given the interval in water level is too small dQ/dz relationship is not valid')
                                self.DQZRN.append((self.QRNout[_ + 1] - self.QRNout[_ - 1]) / (self.ZRNout[_ + 1] - self.ZRNout[_ - 1]))
                            elif _ == self.NumZQout - 1:
                                DZ12 = self.ZRNout[_] - self.ZRNout[_ - 1]
                                if DZ12 <= 1.0E-3:
                                    raise ValueError(f'Given the interval in water level is too small dQ/dz relationship is not valid')
                                self.DQZRN.append((self.QRNout[_] - self.QRNout[_ - 1]) / (self.ZRNout[_] - self.ZRNout[_ - 1]))
                    outlet_factor = True
                except:
                    raise TypeError('The outlet_Q-Z-T should under the standard format!')

            # Import all tributary Discharge - T (Optional)
            elif pd_temp['Time_step'][_ - 1] == 'tribu-Q':

                if pd_temp['Hour'][_] in self.CS_name:
                    tri_cs_name = pd_temp['Time_step'][_]
                    tri_cs_list.append(self.CS_name.index(tri_cs_name))
                else:
                    raise Exception(f"The profile of tribu-Q cross-section {pd_temp['Hour'][_]} is not imported")

                # Derive the tributary discharge - T
                start_index = _ + 1
                end_index = pd_temp.shape[0] if id_index.index(_) == len(id_index) - 1 else id_index[id_index.index(_) + 1] - 1

                try:
                    tri_all_id.append([int(pd_temp['Time_step'][__]) for __ in range(start_index, end_index)])
                    tri_all_time.append([float(pd_temp['Hour'][__]) * 3600 for __ in range(start_index, end_index)])
                    tri_all_q.append([float(pd_temp['Q&Z'][__]) for __ in range(start_index, end_index)])
                except:
                    raise TypeError('The tribu-Z should under the standard format!')

                self.TimeTtby.append(tri_all_time)
                self.TimeQtby.append(tri_all_q)

            # Import all station Q - Z at time 0
            elif pd_temp['Time_step'][_ - 1] == 'init_station_Q-Z':

                # Check if inlet cs is valid
                if station_init_factor:
                    raise Exception('Two init station conditions were imported twice!')

                # Derive the index of CCS in CCS list and CS list
                start_index = _ + 1
                end_index = pd_temp.shape[0] if id_index.index(_) == len(id_index) - 1 else id_index[id_index.index(_) + 1] - 1
                station_list = list(pd_temp['Hour'][start_index: end_index])
                if pd_temp['Hour'][_] == 'Hydrostation':
                    init_ccs_index = [self.Control_CS_station.index(_) for _ in station_list]
                elif pd_temp['Hour'][_] == 'Cross-section':
                    init_ccs_index = [self.Control_CS_Name.index(_) for _ in station_list]
                else:
                    raise Exception('The input station name is not valid')

                # Derive all station Q - Z at time 0
                try:
                    q_list, z_list = [float(__) for __ in pd_temp['Q&Z'][start_index: end_index]], [float(__) for __ in pd_temp['Tributary'][start_index: end_index]]
                    self.Control_CS_Init_Q, self.Control_CS_Init_Z = [np.nan for _ in range(len(self.Control_CS_station))], [np.nan for _ in range(len(self.Control_CS_station))]
                    for ccs_ in init_ccs_index:
                        self.Control_CS_Init_Q[ccs_] = q_list[init_ccs_index.index(ccs_)]
                        self.Control_CS_Init_Z[ccs_] = z_list[init_ccs_index.index(ccs_)]
                    if np.isnan(self.Control_CS_Init_Z[0]) or np.isnan(self.Control_CS_Init_Z[-1]):
                        raise ValueError('The inlet or outlet Z trend is missing')
                    if np.isnan(self.Control_CS_Init_Z[0]) or np.isnan(self.Control_CS_Init_Z[-1]):
                        raise ValueError('The inlet or outlet Q trend is missing')
                    station_init_factor = True
                except:
                    raise TypeError('The init_station_Q-Z should under the standard format!')

            # Distance input discharge - T (Optional)
            elif pd_temp['Time_step'][_ - 1] == 'NFL-Q':
                distance_inflow_factor = True
                raise Exception('The NFL part was not supported yet.')

        # Update the flow parameter
        self.NumTBY = len(self.TimeTtby)
        self.NcsTBY = copy.deepcopy(tri_cs_list)
        self.NumQtby = [len(_) for _ in self.TimeTtby]
        self.NFLDiv = distance_inflow_factor # 引水引沙参数=0(不考虑),=1(考虑)

        # Check the consistency of flow input
        if not inlet_factor:
            raise ValueError('The inlet flow boundary has not successfully imported')
        elif not outlet_factor:
            raise ValueError('The outlet flow boundary has not successfully imported')
        elif not station_init_factor:
            raise ValueError('The initial flow boundary of each station has not successfully imported')
        else:
            if '-T' in self.KBDout and self.TimeTout[1:] != self.TimeTQ[1:]:
                raise ValueError('The time for inlet and outlet flow conditions were not consistent!')

        # Check the consistency of tributary input
        if self.NumTBY > 0:
            if False in [__ == self.TimeTQtby[0] for __ in self.TimeTQtby]:
                raise ValueError('The tributary flow conditions were not consistent!')
            elif self.TimeTQtby[0][1:] != self.TimeTQ[1:]:
                raise ValueError('The tributary flow conditions and inlet time were not consistent!')

        # Linear interpolate the initial water level and discharge to each cross-section
        for cs_ in range(len(self.CS_name)):
            if cs_ > max(self.Control_CS_Id) or cs_ < min(self.Control_CS_Id):
                raise Exception('Code err during the interpolation of flow')
            else:
                try:
                    ccs_id_temp = copy.deepcopy(self.Control_CS_Id)
                    ccs_id_temp.sort()
                    dis_current = self.CS_DistViaRiv[cs_]
                    ccs_q_upstream, ccs_q_downstream, dis_q_upstream, dis_q_downstream = None, None, None, None
                    ccs_z_upstream, ccs_z_downstream, dis_z_upstream, dis_z_downstream = None, None, None, None

                    for ccs_ in range(len(ccs_id_temp)):
                        if (cs_ - ccs_id_temp[ccs_]) * (cs_ - ccs_id_temp[ccs_ + 1]) <= 0:
                            ccs_q_upstream, ccs_q_downstream = ccs_, ccs_ + 1
                            ccs_z_upstream, ccs_z_downstream = ccs_, ccs_ + 1
                            while True:
                                if np.isnan(self.Control_CS_Init_Q[ccs_q_upstream]):
                                    ccs_q_upstream -= 1
                                else:
                                    break
                            while True:
                                if np.isnan(self.Control_CS_Init_Z[ccs_z_upstream]):
                                    ccs_z_upstream -= 1
                                else:
                                    break
                            while True:
                                if np.isnan(self.Control_CS_Init_Q[ccs_q_downstream]):
                                    ccs_q_downstream += 1
                                else:
                                    break
                            while True:
                                if np.isnan(self.Control_CS_Init_Z[ccs_z_downstream]):
                                    ccs_z_downstream += 1
                                else:
                                    break
                            dis_q_upstream, dis_q_downstream = self.CS_DistViaRiv[self.Control_CS_Id[ccs_q_upstream]], self.CS_DistViaRiv[self.Control_CS_Id[ccs_q_downstream]]
                            dis_z_upstream, dis_z_downstream = self.CS_DistViaRiv[self.Control_CS_Id[ccs_z_upstream]], self.CS_DistViaRiv[self.Control_CS_Id[ccs_z_downstream]]
                            break

                    if None in [ccs_q_upstream, ccs_q_downstream, ccs_z_upstream, ccs_z_downstream]:
                        raise Exception('Code err during the linear interpolation!')

                    if dis_q_upstream == 0:
                        self.CSQQ.append(self.Control_CS_Init_Q[ccs_q_upstream])
                    else:
                        self.CSQQ.append(self.Control_CS_Init_Q[ccs_q_upstream] + (self.Control_CS_Init_Q[ccs_q_downstream] - self.Control_CS_Init_Q[ccs_q_upstream]) * (dis_current - dis_q_upstream) / (dis_q_downstream - dis_q_upstream))

                    if dis_z_upstream == 0:
                        self.CSZW.append(self.Control_CS_Init_Z[ccs_z_upstream])
                    else:
                        self.CSZW.append(self.Control_CS_Init_Z[ccs_z_upstream] + (self.Control_CS_Init_Z[ccs_z_downstream] - self.Control_CS_Init_Z[ccs_z_upstream]) * (dis_current - dis_z_upstream) / (dis_z_downstream - dis_z_upstream))
                except:
                    print(traceback.format_exc())
                    raise Exception('Code err during the linear interpolation of initial Q&Z')
        pass

    def _read_roughness_file(self, roughness_file):

        # Read the roughness file
        try:
            pd_temp = pd.read_csv(roughness_file, header=None, encoding='gbk')
        except UnicodeError:
            pd_temp = pd.read_csv(roughness_file, header=None)
        except:
            print(traceback.format_exc())
            raise Exception('Some error occurred during reading the roughness file!')

        # Check if roughness data for all control cross-section is imported
        if len(pd_temp.keys()) - 2 < self.NumCCS:
            raise Exception('The roughness profile of all cross section is not imported')

        # Generate the column name
        columns_t = ['Type', 'var']
        for _ in range(len(pd_temp.keys()) - 2):
            columns_t.append(f'cs_{str(_)}')
        pd_temp.columns = columns_t

        # Generate the control CS roughness list
        self.Control_CS_QDRN = []
        self.Control_CS_Lfloodplain_RN = []
        self.Control_CS_Hfloodplain_RN = []

        # Identify the start index
        id_index = list(pd_temp[pd_temp['Type'] == 'No'].index)
        for _ in id_index:

            # Derive the CS name
            if pd_temp['var'][_ - 1] == 'Hydrostation':
                cs_list = []
                hydro_list = list(pd_temp.iloc[_ - 1, 2:])
                for hydro_ in hydro_list:
                    hydro_factor = False
                    for cs_index in range(len(self.Control_CS_station)):
                        if hydro_ in self.Control_CS_station[cs_index]:
                            cs_list.append(self.Control_CS_Name[cs_index])
                            hydro_factor = True
                            break
                    if not hydro_factor:
                        raise ValueError(f'The CSProf of hydrostation {hydro_} is missing! Please check the _CSProf.csv')

            elif pd_temp['var'][_ - 1] == 'Crosssection':
                cs_list = list(pd_temp.iloc[_ - 1, 2:])
                if False in [__ in self.CS_name for __ in cs_list]:
                    raise Exception(f'The Roughness of {self.CS_name[[__ in self.CS_name for __ in cs_list].index(False)]} is not imported')
            else:
                raise ValueError('The roughness file should clarify the type of section is Hydrostation or cross-section')

            # Generate the range of current
            start_index = _ + 1
            end_index = id_index[id_index.index(_) + 1] - 1 if id_index.index(_) != len(id_index) - 1 else pd_temp.shape[0]

            # Check if the cs list
            if len(cs_list) != len(self.Control_CS_Name):
                raise ValueError('The roughness of control station is not consistent with the control station in CSProf!')
            elif True in [__ not in self.Control_CS_Name for __ in cs_list]:
                raise ValueError(f'The roughness of control station {cs_list[[__ not in self.Control_CS_Name for __ in cs_list].index(True)]} is not control station for CSProf')

            # Read the Qcon-RN
            if str(pd_temp['Type'][_ - 1]) == 'Qcon-RN':

                # Read the RN under different Qcon for each control cross-section
                self.Control_CS_QD = [float(q_) for q_ in list(pd_temp['var'][start_index: end_index])]
                for __ in range(len(self.Control_CS_Name)):
                    try:
                        self.Control_CS_QDRN.append([float(pd_temp[f'cs_{str(cs_list.index(self.Control_CS_Name[__]))}'][___]) for ___ in range(start_index, end_index)])
                    except:
                        raise Exception(f'Non number shown in roughness file for cross section {self.Control_CS_Name[__]}')

                # Check the consistency of Control section Qd and Rn
                for __ in self.Control_CS_QDRN:
                    if len(__) != len(self.Control_CS_QD):
                        raise Exception(f'The QD and RN for control cross section {self.Control_CS_Name[self.Control_CS_QDRN.index(__)]} is not consistent!')

            # Read the FP RN
            elif str(pd_temp['Type'][_ - 1]) == 'FP-RN':

                # Define the FP factor
                FP1_factor, FP2_factor = False, False

                # Get the high and low floodplain roughness
                for index_ in range(start_index, end_index):
                    # Read the RN under different floodplain for each control cross-section
                    if str(pd_temp['var'][index_]) == 'FP1':
                        for __ in range(len(self.Control_CS_Name)):
                            try:
                                self.Control_CS_Hfloodplain_RN.append(float(pd_temp[f'cs_{str(cs_list.index(self.Control_CS_Name[__]))}'][index_]))
                            except:
                                raise Exception(f'The high floodplain roughness file for cross section {self.Control_CS_Name[__]} is missing')
                        FP1_factor = True

                    elif str(pd_temp['var'][index_]) == 'FP2':
                        for __ in range(len(self.Control_CS_Name)):
                            try:
                                self.Control_CS_Lfloodplain_RN.append(float(pd_temp[f'cs_{str(cs_list.index(self.Control_CS_Name[__]))}'][index_]))
                            except:
                                raise Exception(f'The low floodplain roughness file for cross section {self.Control_CS_Name[__]} is missing')
                        FP2_factor = True

                # Check the floodplain factor missed or not
                if not FP1_factor:
                    raise Exception('The roughness for high floodplain were missed!')
                elif not FP2_factor:
                    raise Exception('The roughness for low floodplain were missed!')

                # Check the consistency of Control section LowFP and high FP
                if len(self.Control_CS_Lfloodplain_RN) != len(self.Control_CS_Hfloodplain_RN) or len(self.Control_CS_Lfloodplain_RN) != len(self.Control_CS_Name):
                    raise Exception(f'The high and low floodplain roughness for control cross section is not consistent!')

            elif str(pd_temp['Type'][_ - 1]) == 'DRN-DH':

                # Define the dRN dH factor
                DRN_factor, DH_factor = False, False

                # Get the dRN dH
                for index_ in range(start_index, end_index):
                    if str(pd_temp['var'][index_]) == 'DRN':
                        for __ in range(len(self.Control_CS_Name)):
                            try:
                                self.Control_CS_DNbed.append(float(pd_temp[f'cs_{str(cs_list.index(self.Control_CS_Name[__]))}'][index_]))
                            except:
                                raise Exception(f'The DNbed for cross section {self.Control_CS_Name[__]} is missing')
                        DRN_factor = True

                    elif str(pd_temp['var'][index_]) == 'DH(m)':
                        for __ in range(len(self.Control_CS_Name)):
                            try:
                                self.Control_CS_DHbed.append(float(pd_temp[f'cs_{str(cs_list.index(self.Control_CS_Name[__]))}'][index_]))
                            except:
                                raise Exception(f'The DNbed for cross section {self.Control_CS_Name[__]} is missing')
                        DH_factor = True

                # Check the floodplain factor missed or not
                if not DRN_factor:
                    raise Exception('The DRN profiles were missed!')

                elif not DH_factor:
                    raise Exception('The DG profiles were missed!')

        # Linear interpolate the roughness to each cross-section
        for cs_ in range(len(self.CS_name)):
            if cs_ > max(self.Control_CS_Id) or cs_ < min(self.Control_CS_Id):
                raise Exception('Code err during the interpolation of roughness')
            elif cs_ in self.Control_CS_Id:
                ccs_index = self.Control_CS_Id.index(cs_)
                self.CS_QDRN.append(self.Control_CS_QDRN[ccs_index])
                self.CS_Hfloodplain_RN.append(self.Control_CS_Hfloodplain_RN[ccs_index])
                self.CS_Lfloodplain_RN.append(self.Control_CS_Lfloodplain_RN[ccs_index])
                self.CS_DHbed.append(self.Control_CS_DHbed[ccs_index])
                self.CS_DNbed.append(self.Control_CS_DNbed[ccs_index])
            else:
                ccs_id_temp = copy.deepcopy(self.Control_CS_Id)
                ccs_id_temp.sort()
                dis_current = self.CS_DistViaRiv[cs_]
                ccs_upstream, ccs_downstream, dis_upstream, dis_downstream = None, None, None, None

                for ccs_ in range(len(ccs_id_temp) - 1):
                    if (cs_ - ccs_id_temp[ccs_]) * (cs_ - ccs_id_temp[ccs_ + 1]) < 0:
                        ccs_upstream, ccs_downstream = ccs_, ccs_ + 1
                        ccs_id_upstream, ccs_id_downstream = ccs_id_temp[ccs_], ccs_id_temp[ccs_ + 1]
                        dis_upstream, dis_downstream = self.CS_DistViaRiv[ccs_id_upstream], self.CS_DistViaRiv[ccs_id_downstream]
                        break

                if ccs_upstream is None:
                    raise Exception('Code err during the linear interpolation!')

                self.CS_QDRN.append([self.Control_CS_QDRN[ccs_upstream][qcon_] + (self.Control_CS_QDRN[ccs_downstream][qcon_] - self.Control_CS_QDRN[ccs_upstream][qcon_]) * (dis_current - dis_upstream) / (dis_downstream - dis_upstream) for qcon_ in range(len(self.Control_CS_QD))])
                self.CS_Hfloodplain_RN.append(self.Control_CS_Hfloodplain_RN[ccs_upstream] + (self.Control_CS_Hfloodplain_RN[ccs_downstream] - self.Control_CS_Hfloodplain_RN[ccs_upstream]) * (dis_current - dis_upstream) / (dis_downstream - dis_upstream))
                self.CS_Lfloodplain_RN.append(self.Control_CS_Lfloodplain_RN[ccs_upstream] + (self.Control_CS_Lfloodplain_RN[ccs_downstream] - self.Control_CS_Lfloodplain_RN[ccs_upstream]) * (dis_current - dis_upstream) / (dis_downstream - dis_upstream))
                self.CS_DHbed.append(self.Control_CS_DHbed[ccs_upstream] + (self.Control_CS_DHbed[ccs_downstream] - self.Control_CS_DHbed[ccs_upstream]) * (dis_current - dis_upstream) / (dis_downstream - dis_upstream))
                self.CS_DNbed.append(self.Control_CS_DNbed[ccs_upstream] + (self.Control_CS_DNbed[ccs_downstream] - self.Control_CS_DNbed[ccs_upstream]) * (dis_current - dis_upstream) / (dis_downstream - dis_upstream))

    def _read_bed_material_file(self):
        pass

    def _write_CSProf(self):

        self.input_path = 'G:\\A_1Dflow_sed\\Hydrodynamic_model\\MYR_input\\'
        # Output section starting distance and elevation
        node_list, dis2left_list, elevation_list, type_list = [], [], [], []
        for _ in range(self.Imax):
            node_list.append(f'CS name: {cs_name_list[_]}')
            dis2left_list.append(f'Distance: {self.CS_DistViaRiv[_]} km')
            elevation_list.append(f'Main channel width: {self.BWMC[_]} m')
            type_list.append(f'Mean channel ele: {self.CSZBav[_]} m')

            for __ in range(self.CS_node_num[_]):
                node_list.append(__ + 1)
                dis2left_list.append(self.XXIJ[_][__])
                elevation_list.append(self.ZBIJ[_][__])
                type_list.append(self.KNIJ[_][__])
        output_section_ele = pd.DataFrame({'Column1': node_list, 'Column2': dis2left_list, 'Column3': elevation_list, 'Column4': type_list})
        output_section_ele.to_csv(os.path.join(self.input_path, f'{self.ROI_name}_init_CSProf.csv'), index=False, header=False, encoding='gbk')

        # Output section main channel area starting distance and elevation zxl
        cs_list, DIST_list, BMC_list, ZBk_list, ZBav_list = [], [], [], [], []
        for _ in range(self.Imax):
            cs_list.append(_)
            DIST_list.append(self.CS_DistViaRiv[_])
            BMC_list.append(self.BWMC[_])
            ZBk_list.append(self.ZBBF[_])
            ZBav_list.append(self.CSZBav[_])

        output_section_ele = pd.DataFrame({'I': cs_list, 'DIST(km)': DIST_list, 'BMC(m)': BMC_list, 'ZBk(m)': ZBk_list, 'ZBav(m)': ZBav_list})
        output_section_ele.to_csv(os.path.join(self.input_path, f'{self.ROI_name}_init_CSZBMC.csv'), index=False, header=True, encoding='gbk')

    def _write_CSProf(self, itr):
        pass
        # if no == 0:
        #     fullnam = f"{directory[:dir_lnt]}_CSZBnd.TXT"
        #     with open(fullnam, 'w') as f:
        #         f.write('Each Cross-Sectional Profile in the LYR\n')
        #
        # if int(time_sc) % (24 * 3600) == 0:
        #     with open(fullnam, 'a') as f:
        #         f.write('---------------------------------------------\n')
        #         f.write(f'计算时段 K={i_step:8d} 记录时段 N={no:3d} 总计算时间 Time={time_hr:8.2f} Hour\n')
        #         f.write('---------------------------------------------\n')
        #         for i in range(1, i_max + 1):
        #             f.write(f'第 {i:4d} 断面河底高程\n')
        #             for j in range(1, npt1cs[i - 1] + 1):
        #                 f.write(f'{i:3d} {j:3d} {xxij[i - 1][j - 1]:8.1f} {zbinl[i - 1][j - 1]:8.2f} {zbij[i - 1][j - 1]:8.2f} {zbij[i - 1][j - 1] - zbinl[i - 1][j - 1]:8.3f}\n')
        #
        # fullnam_final = f"{directory[:dir_lnt]}_CSZBnd-FINAL.TXT"
        # with open(fullnam_final, 'w') as f:
        #     f.write('断面地形输入\n')
        #     f.write(f'{i_max:8d}\n')
        #     if time_sc == time_sm * 3600:
        #         for i in range(1, i_max + 1):
        #             f.write(f'{csznum[i - 1]}\n')
        #             f.write('2015年无护岸\n')
        #             f.write(f'{npt1cs[i - 1]}\n')
        #             f.write('序号 起点距 高程\n')
        #             for j in range(1, npt1cs[i - 1] + 1):
        #                 f.write(f'{j} {xxij[i - 1][j - 1]} {zbij[i - 1][j - 1]} {knij[i - 1][j - 1]}\n')
        #
        # if no == 0:
        #     fullnam_avg = f"{directory[:dir_lnt]}_CSZBav.TXT"
        #     with open(fullnam_avg, 'w') as f:
        #         f.write('Lowest and mean bed elevation in the LYR\n')
        #
        # if int(time_sc) % (24 * 3600) == 0:
        #     with open(fullnam_avg, 'a') as f:
        #         f.write(f'计算时段 K={i_step:4d} 记录时段 N={no:3d} 总计算时间 Time={time_hr:8.2f} Hour\n')
        #         f.write('i Dist ZBmin ZBav\n')
        #         for i in range(1, i_max + 1):
        #             f.write(f'{i:4d} {distlg[i - 1]:8.2f} {cszbmn[i - 1]:10.2f} {cszbav[i - 1]:10.2f}\n')

    def _write_BedProf(self, itr):
        pass
        # # Call to compute size of bed material
        # compsizebm()
        #
        # if No == 0:  # Initial time file format
        #     fullnam = f"{Directory[:Dirlnt]}_BMGrad.TXT"
        #     with open(fullnam, 'w') as file:
        #         file.write('Median and Mean Diameters of Bed Material in LYR\n')
        #
        # if Istep == 0 or Istep == Nstep:
        #     # If mod(int(TimeSC), 24*3600) == 0:  # Record results every 24 hours
        #     with open(fullnam, 'a') as file:
        #         file.write('---------------------------------------------------\n')
        #         file.write(f'计算时段 K={Istep:8d} 记录时段 N={No:3d} 总计算时间 Time={TimeHR:8.2f} Hour\n')
        #         file.write('---------------------------------------------------\n')
        #         file.write('断面号 记忆层数 床沙中值粒径(mm) 床沙平均粒径(mm)\n')
        #         for I in range(1, Imax + 1):
        #             file.write(f'{I:4d} {NTPML[I]:7d} {BED50[I] * 1000.0:10.4f} {BEDPJ[I] * 1000.0:9.4f}\n')

    def _init_flow_boundary(self):

        # Initiate the flow boundary
        for _ in range(len(self.CSQQ)):
            AAi, BBi, DMKi, alfi = compute_channel_char(self.Hmin, self.DHmin, self.DBIJ[_], self.ZBIJ[_], self.KNIJ[_], self.RNIJ[_],  self.CSZW[_])

            if AAi > 0.0:
                self.CSUM.append(self.CSQQ[_]/AAi)
                self.CSHM.append(AAi / BBi)
            else:
                self.CSUM.append(0.0)
                self.CSHM.append(self.Hmin)

        # Initiate the bed deform para
        if self.BedDeform_flag:
            self.DZMC = [0 for _ in range(self.CS_num)] # 主槽最大冲刷厚度(与初始时刻相比)
            self.DSDt = [0 for _ in range(self.CS_num)] # 含沙量大小随时间变化项
            self.DASDt1 = [0 for _ in range(self.CS_num)]
        else:
            self.DZMC = [np.nan for _ in range(self.CS_num)] # 主槽最大冲刷厚度(与初始时刻相比)
            self.DSDt = [np.nan for _ in range(self.CS_num)] # 含沙量大小随时间变化项
            self.DASDt1 = [np.nan for _ in range(self.CS_num)]

        # Initiate the flow para
        self.update_flow_para(self.CSZW, self.CSQQ)

    def _compute_rn4channel_floodplain_method1(self):

        self.RNIJ = []
        for _ in range(len(self.CS_QDRN)):

            # Get the Q-RN for the cross-section _
            Q_ = copy.deepcopy(self.Control_CS_QD)
            Q_.sort()
            RN = copy.deepcopy(self.CS_QDRN[_])

            # Linear interpolate the roughness of main channel at discharge CSQQ
            RNIJ = []
            RNmain = None
            for q_ in range(len(Q_) - 1):
                if (Q_[q_] - self.CSQQ[_]) * (Q_[q_ + 1] - self.CSQQ[_]) < 0:
                    rn_up, q_up, rn_low, q_low = RN[self.Control_CS_QD.index(Q_[q_ + 1])], Q_[q_ + 1], RN[self.Control_CS_QD.index(Q_[q_])], Q_[q_]
                    RNmain = rn_low + (rn_up - rn_low) * (self.CSQQ[_] - q_low) / (q_up - q_low)
                elif self.CSQQ[_] < min(Q_):
                    RNmain = RN[self.Control_CS_QD.index(min(Q_))]
                    break
                elif self.CSQQ[_] > max(Q_):
                    RNmain = RN[self.Control_CS_QD.index(max(Q_))]
                    break

            # Adjust the roughness after bed deformation
            if self.BedDeform_flag:
                RNmain = RNmain + self.DZMC[_] * (self.DNBDCS[_] / self.DHBDCS[_])
                RNmain = min(0.060, RNmain)
                RNmain = max(0.008, RNmain)

            # Reassign the roughness based on the discharge
            if RNmain is None:
                raise Exception('Code error DURING THE roughness computation')
            else:
                for __ in range(self.CS_node_num[_]):
                    if self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3:
                        RNIJ.append(RNmain)
                    elif self.KNIJ[_][__] == 1:
                        RNIJ.append(self.CS_Lfloodplain_RN[_])
                    elif self.KNIJ[_][__] == 2:
                        RNIJ.append(self.CS_Lfloodplain_RN[__])
            self.RNIJ.append(RNIJ)

    def _compute_rn4channel_floodplain_method2(self):

        # Calculate the main channel roughness
        self.RNIJ = []
        self._compute_rn4channel()

        # Reassign the roughness
        for _ in range(len(self.CS_QDRN)):
            RNQ = self.CS_QDRN[_]
            RNIJ = []
            for __ in range(self.CS_node_num[_]):
                if self.KNIJ[_][__] == 0 or self.KNIJ[_][__] == 3:
                    RNIJ.append(self.RNMC[_])
                elif self.KNIJ[_][__] == 1:
                    RNIJ.append(self.CS_Lfloodplain_RN[_])
                elif self.KNIJ[_][__] == 2:
                    RNIJ.append(self.CS_Lfloodplain_RN[__])

    def _compute_rn4channel(self):

        # Compute the roughness 4 channel
        self.RNMC = []
        for _ in range(self.CS_num):

            # Compute the Froude number
            if self.CSUM[_] > self.Hmin:
                Fr = self.CSUM[_] / np.sqrt(self.Grav * self.CSHM[_])
                Fr = max(0.10, Fr)
                Fr = min(0.80, Fr)
            else:
                Fr = 0.001

            # Using Zhang HW Equation to compute roughness (赵连军，张红武．黄河下游河道水流摩阻特性的研究 ［J］．人民黄河1997（9）)
            if self.MDRGH == 2:
                if self.CSUM[_] > self.Hmin:
                    DELT = self.BED50[_] * 10 ** (10 * (1.0 - np.sqrt(np.sin(np.pi * Fr))))
                    CN = 0.15 * (1.0 - 4.2 * np.sqrt(self.CSSVL[_]) * (0.365 - self.CSSVL[_]))
                    DELTH = DELT / self.CSHM[_]
                    DELTH = min(0.8, DELTH)

                    RN1 = (CN * DELT) / (np.sqrt(self.Grav) * self.CSHM[_] ** (5/6))
                    RN2 = 0.49 * DELTH ** 0.77 + 0.375 * np.pi * (1.0 - DELTH) * (np.sin(DELTH ** 0.2) ** 5.0)

                    RM = RN1 / RN2
                    RM = min(0.04, RM)
                    RM = max(0.01, RM)
                    self.RNMC.append(RM)
                else:  # Water depth less than virtual water depth
                    self.RNMC.append(0.040)

                if self.CSUM[_] <= 0.00010:  # Very low flow velocity
                    self.RNMC.append(0.040)

            elif self.MDRGH == 3:  # Using other EQ
                RM = 0.005 * (Fr ** -0.8939)
                RM = min(0.04, RM)
                RM = max(0.01, RM)
                self.RNMC.append(RM)

            elif self.MDRGH == 4:  # Using Liu Xin EQ
                if self.CSHH[_] > self.Hmin:
                    RM = 1.2 * 0.29 * self.CSHH[_] ** (1.0 / 6.0) / np.sqrt(8.0 * self.Grav) * Fr ** (-0.893) * (self.CSHH[_] / self.BED50[_]) ** (-0.24)
                    RM = min(0.06, RM)
                    RM = max(0.008, RM)
                    self.RNMC.append(RM)
                else:  # Water depth less than virtual water depth
                    self.RNMC.append(0.060)

                if self.CSUM[_] <= 0.00010:  # Very low flow velocity
                    self.RNMC.append(0.060)

    def update_flow_para(self, Z, Q):

        # Reassign the Z and Q into the model
        self.CSZW = copy.deepcopy(Z)
        self.CSQQ = copy.deepcopy(Q)

        # Calculate the roughness
        if self.MDRGH == 4: # Liu Xin Eq to calculate the roughness
            self._compute_rn4channel_floodplain_method2()

        if self.Istep == 1 and self.SPTime < 2.0 * self.DTstep:
            self._compute_rn4channel_floodplain_method2()

        # Calculate the para
        for i in range(self.CS_num):
            BBJ, QQJ = [], []  # 各子断面的过水宽度/ 过流流量
            HHJ, WPJ, RDJ = [], [], []  # 各子断面的平均水深/ 湿周/ 水力半径
            RNJ, UUJ, UTJ = [], [], []  # 各子断面的平均糙率/ 流速/ 摩阻流速
            AAJ, QKJ = [], []  # 各子段面过水面积/ 流量模数
            HCJ = []  # 子段面的形心
            HH = compute_water_depth(self.ZBIJ[i], self.KNIJ[i], self.CSZW[i], self.Hmin)

            # Compute the hydraulic para of subsections
            for _ in range(len(HH) - 1):

                # Right trapezoid subsection
                if HH[_] > self.Hmin and HH[_ + 1] > self.Hmin:
                    BBJ.append(self.DBIJ[i][_])
                    HHJ.append((HH[_] + HH[_ + 1]) / 2)
                    RNJ.append((self.RNIJ[i][_] + self.RNIJ[i][_ + 1]) / 2)
                    WPJ.append(np.sqrt(BBJ[-1] ** 2 + (HH[_] + HH[_ + 1]) ** 2))
                    AAJ.append(HHJ[-1] * BBJ[-1])
                    RDJ.append(AAJ[-1] / WPJ[-1])
                    QKJ.append(AAJ[-1] * (RDJ[-1] ** (2 / 3)) / RNJ[-1])

                    a0 = HH[_]
                    b0 = HH[_ + 1]
                    h0 = BBJ[-1]

                # Non-flooded subsection
                elif HH[_] <= self.Hmin and HH[_ + 1] <= self.Hmin:
                    BBJ.append(0.0)
                    HHJ.append(0.0)
                    RNJ.append((self.RNIJ[i][_] + self.RNIJ[i][_ + 1]) / 2)
                    WPJ.append(0.0)
                    AAJ.append(0.0)
                    RDJ.append(0.0)
                    QKJ.append(0.0)

                    a0 = 0.0
                    b0 = 0.0
                    h0 = 0.0

                # Right triangular subsection
                elif HH[_] <= self.Hmin and HH[_ + 1] > self.Hmin:  # CASE-3
                    dx = self.DBIJ[i][_]
                    hh_ = self.ZBIJ[i][_] - self.CSZW[i]
                    hh_ = 0.0 if hh_ <= self.Hmin else hh_

                    BBJ.append(dx * HH[_ + 1] / (hh_ + HH[_ + 1]))
                    HHJ.append(0.5 * HH[_ + 1])
                    RNJ.append((self.RNIJ[i][_] + self.RNIJ[i][_ + 1]) / 2)
                    WPJ.append(np.sqrt(BBJ[-1] ** 2.0 + HH[_ + 1] ** 2.0))
                    AAJ.append(HHJ[-1] * BBJ[-1])
                    RDJ.append(AAJ[-1] / WPJ[-1])
                    QKJ.append(AAJ[-1] * (RDJ[-1] ** (2 / 3)) / RNJ[-1])

                    a0 = 0.0
                    b0 = HH[_ + 1]
                    h0 = BBJ[-1]

                # Right triangular subsection
                elif HH[_] > self.Hmin and HH[_ + 1] <= self.Hmin:  # CASE-4
                    dx = self.DBIJ[i][_]
                    hh_ = self.ZBIJ[i][_ + 1] - self.CSZW[i]
                    hh_ = 0.0 if hh_ <= self.Hmin else hh_

                    BBJ.append(dx * HH[_] / (hh_ + HH[_]))
                    HHJ.append(0.5 * HH[_])
                    RNJ.append((self.RNIJ[i][_] + self.RNIJ[i][_ + 1]) / 2)
                    WPJ.append(np.sqrt(BBJ[-1] ** 2.0 + HH[_] ** 2.0))
                    AAJ.append(HHJ[-1] * BBJ[-1])
                    RDJ.append(AAJ[-1] / WPJ[-1])
                    QKJ.append(AAJ[-1] * (RDJ[-1] ** (2 / 3)) / RNJ[-1])

                    a0 = HH[_]
                    b0 = 0
                    h0 = BBJ[-1]

                else:
                    raise Exception('Code err during the calculation of hydraulic para')

                # Calculate centroid of subsection
                if (a0 + b0) > 0.0:
                    HCJ.append((a0 ** 2 + b0 ** 2 + a0 * b0) / (a0 + b0) / 3.0)
                else:
                    HCJ.append(0.0)

            # Sum the para of subsection, and get the para of each cross-section
            CSAA = sum(AAJ)  # 过水面积
            CSQK = sum(QKJ)  # 流量模数
            CSBB = sum(BBJ)  # 水面宽度
            CSWP = sum(WPJ)  # 湿周
            ahc = sum([HCJ[j] * AAJ[j] for j in range(len(AAJ))])

            CSHH = CSAA / CSBB if CSBB > 0 else 0  # 水深
            CSRD = CSAA / CSWP if CSWP > 0 else 0  # 水力半径
            CSHC = ahc / CSAA if CSAA > 0 else 0  # 形心下水深
            CSZB = self.CSZW[i] - CSHH  # 平均河底高程

            CSUU = self.CSQQ[i] / CSAA if CSAA > 0 else 0  # 流速
            CSSP = (self.CSQQ[i] ** 2) / (CSQK ** 2) if CSQK > 0 else 0  # 能坡
            CSRN = (CSRD ** (2 / 3)) * (CSSP ** 0.5) / CSUU if CSUU > 0 else 0  # 糙率
            CSUT = (self.Grav ** 0.5) * CSRN * CSUU / (CSRD ** (1.0 / 6.0))  # 摩阻流速

            # Calculate the hydraulic feature for MAIN CHANNEL and FLOODPLAIN
            CSQM, CSAM, CSBM = 0.0, 0.0, 0.0
            CSQF, CSAF, CSBF = 0.0, 0.0, 0.0
            CSBD = 0.0
            UUJ, UTJ, QQJ = [], [], []
            for _ in range(len(HH) - 1):
                if HH[_] > 0.0:
                    UUJ.append((RDJ[_] ** (2.0 / 3.0)) * (CSSP ** 0.5) / RNJ[_])
                    UTJ.append((self.Grav ** 0.5) * RNJ[_] * UUJ[-1] / (RDJ[_] ** (1.0 / 6.0)))
                else:
                    UUJ.append(0.0)
                    UTJ.append(0.0)

                QQJ.append(self.CSQQ[_] * (QKJ[_] / CSQK))

                if self.KNIJ[i][_] * self.KNIJ[i][_ + 1] != 2 and self.KNIJ[i][_] * self.KNIJ[i][_ + 1] != 4:
                    CSQM += QQJ[-1]
                    CSAM += AAJ[-1]
                    CSBM += BBJ[-1]

                if self.KNIJ[i][_] * self.KNIJ[i][_ + 1] == 0:
                    CSBD += BBJ[-1]

                if self.KNIJ[i][_] * self.KNIJ[i][_ + 1] == 3 or self.KNIJ[i][_] * self.KNIJ[i][_ + 1] == 9:
                    if tzbij[_] > tzbinl[_]:
                        CSBD += BBJ[-1]

                if self.KNIJ[i][_] == 1 or self.KNIJ[i][_] == 2:
                    CSQF += QQJ[-1]
                    CSAF += AAJ[-1]
                    CSBF += BBJ[-1]

            CSHM = CSAM / CSBM if CSBM > 0.0 else 0.0
            CSHF = CSAF / CSBF if CSBF > 0.0 else 0.0
            CSUM = CSQM / CSAM if CSAM > 0.0 else 0.0
            CSUF = CSQF / CSAF if CSAF > 0.0 else 0.0

            # Calculate Froude number
            sumka = sum([QKJ[j] ** 3.0 / AAJ[j] ** 2.0 if AAJ[j] > 0.0 else 0.0 for j in range(len(AAJ))])
            if CSAA > 0.0:
                CSALF = sumka / (CSQK ** 3.0 / CSAA ** 2.0)
            else:
                CSALF = 1.0

            if CSAA > 0.0:
                CSFR = CSUU * (CSALF ** 0.5) / (self.Grav * CSHH) ** 0.5
            else:
                CSFR = 1.01

            # Calculate the hydraulic feature for each node
            Unode, Hnode, Qnode, UTnode, TFnode = [], [], [], [], []
            for _ in range(len(HH)):
                Hnode.append(HH[_])
                if HH[_] > self.Hmin:
                    Unode.append((Hnode[-1] ** (2.0 / 3.0)) * (CSSP ** 0.5) / self.RNIJ[i][_])
                    Qnode.append(Hnode[-1] * Unode[-1])
                    UTnode.append((self.Grav ** 0.5) * self.RNIJ[i][_] * Unode[-1] / (Hnode[-1] ** (1.0 / 6.0)))
                    TFnode.append((UTnode[-1] ** 2.0) * 1000.0)
                else:
                    Unode.append(0.0)
                    Qnode.append(0.0)
                    UTnode.append(0.0)
                    TFnode.append(0.0)

    def run_model(self):

        # Run the 1D model
        print('------- Run the 1D Hydrodynamics model -------')

        # Initialize parameter
        self.Istep = 0
        self.NumRem = 0  # Number of times to calculate water and sediment conditions and topography changes
        self.TimeSC = 0.0  # Total calculation time (seconds)
        self.TimeHR = 0.0  # Total calculation time (hours)

        # Input initial cross-section flow and water level
        self._init_flow_boundary()

        # Write the initial CSProfile and Bed degradation
        if self.BedDeform_flag and self.SedTrans_flag:
            print('------- The sediment transport model will be executed -------')
            print("------- Write the OutputCSProfiles -------")
            self._write_CSProf(self.NumRem)
            print("-------    Write the BedProfiles   -------")
            self._write_BedProf(self.NumRem)
            print("-------  Setup the spin-up period  -------")
            if self.NYspin:
                NFLBDtp = self.BedDeform_flag
                NFLGAtp = self.GradAdj_flag
                SPtime = 0.0  # Initial simulation time

        # Iterate the Nstep
        for self.Istep in range(1, self.Nstep):

            # Code for the spin-up period
            if self.NYspin == 1:
                sptime += dtstep / 3600.0

            if self.NYspin == 1 and sptime <= timesp:
                nflbd = 0
                nflga = 0
                pleft = 100.0 - 100 * sptime / timesp
                print(f'Spin-up Period, SPtime,Left= {sptime} {pleft}%')

            if nyspin == 1 and sptime > timesp:
                nflbd = nflbdtp
                nflga = nflgatp

            nflrd = 0
            timesc = dtstep * istep  # Accumulated calculation time (Second)
            timehr = timesc / 3600.0  # Accumulated calculation time (Hour)

            # Determine whether to record calculation results
            if istep % ntrd == 0:
                numrem += 1  # Number of result records
                nflrd = 1  # Result recording parameter

            # Determine boundary conditions for one-dimensional flow calculation
            print('Hydromodel ---- Start computing 1D flow boundary ------')
            self.compute_1D_flowbound()  # Interpolate flow boundary conditions to the calculation time
            print('Hydromodel ---- Finish computing 1D flow boundary ------')

            # Calculate one-dimensional constant water surface line (considering tributary inflow and outflow)
            print('Hydromodel ---- Start computing 1D constant water surface line ------')
            self.solve_1D_flow_RT()  # Push water surface line
            print('Hydromodel ---- Finish computing 1D constant water surface line ------')

            # Record the hydro condition

    def compute_1D_flowbound(self):

        # Calculate the Discharge, roughness and water level along the channel at the time self.TimeSC
        TTT = []
        QQQ = []

        # Take Flow at Time 0 as BOUNDARY
        TimeSC0 = self.TimeSC
        if self.NYspin == 1 and self.SPtime <= self.TimeSP:
            TimeSC0 = 0.0  #

        # Calculate the minimum bed elevation at the outlet
        Zbout = 10000.0
        for _ in range(self.CS_node_num[-1]):
            if (self.KNIJ[-1][_] == 0 or self.KNIJ[-1][_] == 3) and self.ZBIJ[-1][_] <= Zbout:
                Zbout = self.ZBIJ[-1][_]

        # Interpolate the inlet Q and T to TimeSC0
        self.Qinlet_new = linear_interpolation(self.TimeTQ, self.TimeQint, TimeSC0)
        WTemp = linear_interpolation(self.TimeTMP, self.TimeWTemp, TimeSC0)

        # Interpolate the outlet Q, Z, T to TimeSC0
        if self.KBDout == 1:  # t-Q Relation
            self.Qoutlet_new = linear_interpolation(self.TimeTout, self.TimeQout,  TimeSC0)
        elif self.KBDout == 2:  # t-Z Relation
            self.Zoutlet_new = linear_interpolation(self.TimeTout, self.TimeZout, TimeSC0)
            if self.Zoutlet_new <= Zbout:
                self.Zoutlet_new = Zbout  # 防止水位过低
        elif self.KBDout == 3:  # Z-Q Relation
            self.DQDZoutlet_new = cubic_spline_interpolation(self.ZRNout, self.DQZRN, self.CSZW[-1])  # 出口处水位流量关系的导数

        # Calculate the lateral input
        if self.NYspin == 0:  # For the compute period
            for _ in range(self.CS_num - 1):
                if self.Istep == 1:
                    self.QLold[_] = 0.0
                else:
                    self.QLold[_] = self.QL_new[_]
                self.QL_new[_] = 0.0  # 侧向出入流量(m3/s.m)

        elif self.NYspin == 1:  # For the Spin-up period
            for _ in range(self.CS_num - 1):
                self.QLold[_] = self.QL_new[_]
                self.QL_new[_] = 0.0

        # Tributary
        if self.NumTBY > 0:  # 存在支流入汇(+)或流出(-)
            for _ in range(self.NumTBY):
                for k in range(self.NumQtby[_]):
                    TTT[k] = self.TimeTtby[_][k]
                    QQQ[k] = self.TimeQtby[_][k]  # No-时间-支流N的 流量

                QL = linear_interpolation(TTT, QQQ, TimeSC0)  # m3/s
                ICS = self.NcsTBY[_]  # 支流支流所在河段的断面区间号码
                self.QL_new[ICS] = self.QL_new[ICS] + QL / self.DX2CS[ICS]  # m3/s.m

        #  在区间考虑沿程引水
        if self.NFLDIV > 0:  # 存在沿程引水
            for i in range(self.CS_num - 1):
                for k in range(self.NumTdiv):
                    QQQ[k] = self.TimeWdiv[k][i]  # m3/s.m   两断面间的引水流量(单位长度)

                QL = linear_interpolation(self.TimeTdiv, QQQ, TimeSC0)  # m3/s.m
                self.QL_new[i] = self.QL_new[i] - QL  # m3/s.m   沿程引水取负值

        # Compute the roughness of each subsection
        if self.MDRGH == 1:
            self._compute_rn4channel_floodplain_method1()  # Using Q-N Relation
        elif self.MDRGH in [2, 3, 4]:
            self._compute_rn4channel_floodplain_method2()  # Using ZhangHW Eq or other EQ

    def solve_1D_flow_RT(self):

        # Get the Qold Zold
        Qold, Zold = copy.deepcopy(self.CSZW), copy.deepcopy(self.CSQQ)

        # Retrieve the A B DMK ALF at current time
        Aold, Bold, DMKold, ALFold = [], [], [], []

        for _ in range(0, self.Imax):
            ZWL = Zold[_]
            NMC = self.NMC1CS[_]  # Number of main channels in the section
            AAi, BBi, DMKi, alfi = compute_channel_char(self.Hmin, self.DHmin, self.DBIJ[_], self.ZBIJ[_], self.KNIJ[_], self.RNIJ[_], self.CSZW[_])
            Aold.append(AAi)
            Bold.append(BBi)
            DMKold.append(DMKi)
            ALFold.append(alfi)

        # Calculate the Qnew Znew
        Qnew, Znew = self.PreissmannScheme(Qold, Zold, Bold, Aold, DMKold, ALFold)

        # Update the hydraulic para
        self.update_flow_para(Znew, Qnew)

        #
        for _ in range(self.Imax):
            self.DAFDT[_] = (self.CSAA[_] - Aold[_]) / self.DTstep

    def PreissmannScheme(self, Qold, Zold, Bold, Aold, DMKold, ALFold):

        # Initialize arrays for calculations
        Qnew, Znew = [0.0 for _ in range(self.Imax)], [0.0 for _ in range(self.Imax)]

        # Arrays for hydraulic parameters at the mid-point
        DADXmid, QLmid, Qmid, Zmid, Amid, Bmid, DMKmid, ALFmid = [], [], [], [], [], [], [], []

        # additional terms in momentum and continuity equations
        Add_ContinueEq, Add_momentumEq = [], []

        # 中间点出的侧向来流
        for _ in range(self.Imax - 1):
            QLmid.append(self.CitaFW * self.QL_new[_] + (1.0 - self.CitaFW) * self.QLold[_])

        for itr in range(self.ITSUMF):
            if itr == 1:
                Qnew_itr = copy.deepcopy(Qold)
                Znew_itr = copy.deepcopy(Zold)
            else:
                Qnew_itr = copy.deepcopy(Qnew)
                Znew_itr = copy.deepcopy(Znew)

            # Compute hydraulic para at the mid-point
            Anew, Bnew, DMKnew, ALFnew, DADXmid = self.comp_qzba2(Zold, Znew_itr)

            for _ in range(self.Imax - 1):
                Amid.append(0.5 * self.CitaFW * (Anew[_ + 1] + Anew[_]) + 0.5 * (1.0 - self.CitaFW) * (Aold[_ + 1] + Aold[_]))
                Bmid.append(0.5 * self.CitaFW * (Bnew[_ + 1] + Bnew[_]) + 0.5 * (1.0 - self.CitaFW) * (Bold[_ + 1] + Bold[_]))
                DMKmid.append(0.5 * self.CitaFW * (DMKnew[_ + 1] + DMKnew[_]) + 0.5 * (1.0 - self.CitaFW) * (DMKold[_ + 1] + DMKold[_]))
                ALFmid.append(0.5 * self.CitaFW * (ALFnew[_ + 1] + ALFnew[_]) + 0.5 * (1.0 - self.CitaFW) * (ALFold[_ + 1] + ALFold[_]))
                Qmid.append(0.5 * self.CitaFW * (Qnew_itr[_ + 1] + Qnew_itr[_]) + 0.5 * (1.0 - self.CitaFW) * (Qold[_ + 1] + Qold[_]))

            # Compute additional terms in momentum and continuity equations
            if not self.NCPFS or not self.SedTrans_flag:
                Add_ContinueEq = [0 for _ in range(self.Imax - 1)]
                Add_momentumEq = [0 for _ in range(self.Imax - 1)]
            elif self.NCPFS and self.SedTrans_flag:
                Dp = self.Psedi - self.Pflow
                P0 = (1.0 - self.Pdry / self.Psedi) * self.Pflow + self.Pdry

                for i in range(self.Imax - 1):
                    pm = 0.5 * (self.CSPM[i] + self.CSPM[i + 1])
                    Hcm = 0.5 * (self.CSHC[i] + self.CSHC[i + 1])
                    dsdtm = 0.5 * (self.dSdt[i] + self.dSdt[i + 1])
                    dsdxm = (self.CSSUS[i + 1] - self.CSSUS[i]) / self.DX2CS[i]
                    DA0dtm = 0.5 * (self.DASDt1[i] + self.DASDt1[i + 1])

                    # Compute Adtm terms
                    Adtm1 = 0.0
                    Adtm2 = -(Dp * Qmid[i] * dsdtm) / (pm * self.Psedi)  # From md1
                    Adtm3 = -(Dp * (Qmid[i] ** 2 / Amid[i] + self.Grav * Hcm * Amid[i]) * dsdxm) / (pm * self.Psedi)

                    # Optionally modify Adtm2 and Adtm3 based on comments
                    # Uncomment and adjust if necessary
                    # Adtm2 += parameters['CSUU'][i] * (P0 / pm) * DA0dtm
                    # Adtm3 -= (Dp * grav * Hcm * Amid[i] * dsdxm) / (pm * Psedi)

                    Add_momentumEq.append(Adtm1 + Adtm2 + Adtm3)
                    Add_ContinueEq.append(-DA0dtm)

            # 计算连续方程离散后的系数矩阵
            aa1, bb1, cc1, dd1, ee1 = [], [], [], [], []
            for _ in range(self.Imax - 1):
                aa1.append(0.5 * Bmid[_] / self.DTstep)
                bb1.append(-self.CitaFW / self.DX2CS[_])
                cc1.append(aa1[_])
                dd1.append(-bb1[_])
                ee1.append(QLmid[_] - (1.0 - self.CitaFW) * (Qold[_ + 1] - Qold[_]) / self.DX2CS[_] + 0.5 * Bmid[_] * (Zold[_ + 1] + Zold[_]) / self.DTstep + Add_ContinueEq[_])

            # 计算动量方程离散后的系数矩阵
            aa2, bb2, cc2, dd2, ee2 = [], [], [], [], []
            for _ in range(self.Imax - 1):
                CF1 = self.Grav * Amid[_] - ALFmid[_] * Bmid[_] * Qmid[_] ** 2.0 / Amid[_] ** 2.0
                CF2 = 2.0 * ALFmid[_] * Qmid[_] / Amid[_]
                CF3 = 2.0 * self.DTstep * self.CitaFW / self.DX2CS[_]

                aa2.append(-CF1 * CF3)
                bb2.append(1.0 - CF2 * CF3)
                cc2.append(CF1 * CF3)
                dd2.append(1.0 + CF2 * CF3)

                e21 = Qold[_ + 1] + Qold[_]
                e22 = -2.0 * self.DTstep * CF1 * (1.0 - self.CitaFW) * (Zold[_ + 1] - Zold[_]) / self.DX2CS[_]
                e23 = -2.0 * self.DTstep * CF2 * (1.0 - self.CitaFW) * (Qold[_ + 1] - Qold[_]) / self.DX2CS[_]
                e24 = (2.0 * self.DTstep) * 0.0
                e25 = 2.0 * self.DTstep * ((Qmid[_] / Amid[_]) ** 2.0) * DADXmid[_]
                e26 = -(2.0 * self.DTstep) * self.Grav * Amid[_] * abs(Qmid[_]) * Qmid[_] / (DMKmid[_] ** 2.0)

                uup = Qold[_] / Aold[_]
                ulw = Qold[_ + 1] / Aold[_ + 1]
                cof = compute_cof(Bold[_], Bold[_ + 1])
                e27 = -self.DTstep * Amid[_] * cof * abs(uup ** 2.0 - ulw ** 2.0) / self.DX2CS[_]

                ee2.append(e21 + e22 + e23 + e24 + e25 + e26 + e27 + Add_momentumEq[_] * 2.0 * self.DTstep)

            # 确定来流过程
            if self.Qinlet_new:
                raise Exception('Code err the inlet flow boundary is not computed!')

            PPP, RRR, Qnew = [copy.deepcopy(self.Qinlet_new)], [0.0], [copy.deepcopy(self.Qinlet_new)]
            for _ in range(self.Imax - 1):
                Abr1 = aa1[_] + bb1[_] * RRR[_]
                Abr2 = aa2[_] + bb2[_] * RRR[_]
                RR1 = -cc2[_] * Abr1 + cc1[_] * Abr2
                RR2 = dd2[_] * Abr1 - dd1[_] * Abr2

                if abs(RR2) <= 0.0001:
                    print(f'CS=i,i+1: {str(_)}, {str(_ + 1)}')
                    print('rr2.le.0.0')
                    print(f'DXcs= {str(self.DX2CS[_] / 1000.0)} km')
                    input("Press Enter to continue...")

                RRR.append(RR1 / RR2)
                PP1 = (ee2[_] - bb2[_] * PPP[_]) * Abr1 - (ee1[_] - bb1[_] * PPP[_]) * Abr2
                PP2 = RR2
                PPP.append(PP1 / PP2)

            # 确定出口流量过程
            # Q-T outlet relationship
            if self.KBDout == 1:
                if self.Qoutlet_new == 0:
                    raise Exception('Code err the outlet flow boundary is not computed!')
                Qnew[self.Imax] = copy.deepcopy(self.Qoutlet_new)
                if abs(RRR[self.Imax]) > 1.0E-6:
                    Znew[self.Imax] = (self.Qoutlet_new - PPP[self.Imax]) / RRR[self.Imax]
                else:
                    print(f'RRR(imax)=0 {str(RRR[self.Imax])}')
                    input("Press Enter to continue...")

            # Z-T outlet relationship
            if self.KBDout == 2:
                if self.Zoutlet_new == 0:
                    raise Exception('Code err the outlet flow boundary is not computed!')
                Znew[self.Imax] = copy.deepcopy(self.Zoutlet_new)
                Qnew[self.Imax] = PPP[self.Imax] + RRR[self.Imax] * copy.deepcopy(self.Zoutlet_new)

            # Case 3: Determination of the Downstream boundary
            if self.KBDout == 3:  # Z-Q Relation
                if self.DQDZoutlet_new == 0:
                    raise Exception('Code err the outlet flow boundary is not computed!')
                DZNout = (PPP[self.Imax] - Qold[self.Imax] + RRR[self.Imax] * Zold[self.Imax]) / (self.DQDZoutlet_new - RRR[self.Imax])
                Znew[self.Imax] = Zold[self.Imax] + DZNout
                Qnew[self.Imax] = Qold[self.Imax] + self.DQDZoutlet_new * DZNout

                if Znew[self.Imax] < self.CSZBmn[self.Imax]:
                    Znew[self.Imax] = self.CSZBmn[self.Imax] + self.DHmin
                    Qnew[self.Imax] = self.Qmin

            # 回溯计算水位
            for _ in range(self.Imax - 1, 0, -1):
                DZ11 = (ee2[_] - bb2[_] * PPP[_]) * dd1[_] - (ee1[_] - bb1[_] * PPP[_]) * dd2[_]
                DZ12 = dd1[_] * (aa2[_] + bb2[_] * RRR[_]) - dd2[_] * (aa1[_] + bb1[_] * RRR[_])

                if abs(DZ12) <= 0.0001:
                    print(f'CS=i,i+1 {str(_)} {str(_ + 1)}')
                    print(f'DX2cs= {str(self.DX2CS[_] / 1000.0)} km')
                    input("Press Enter to continue...")

                DZ1 = DZ11 / DZ12
                DZ21 = -cc2[_] * dd1[_] + cc1[_] * dd2[_]
                DZ22 = DZ12
                DZ2 = DZ21 / DZ22
                Znew[_] = DZ1 + DZ2 * Znew[_ + 1]

                # Do not calculate the flow increment at the inlet boundary
                if _ != 0:
                    Qnew[_] = PPP[_] + RRR[_] * Znew[_]

            # Calculate the itr error
            errQ, errZ, MDQZ = 0.0, 0.0, 1
            
            # Process each point
            for _ in range(self.Imax):
                if MDQZ == 1:  # Relative value
                    if abs(Qnew[_]) > 1.0E-3:
                        erq = abs((Qnew_itr[_] - Qnew[_]) / Qnew[_])
                    else:
                        erq = 0.0

                    Hnew = Znew[_] - self.CSZBmn[_]
                    Hnewt = Znew_itr[_] - self.CSZBmn[_]

                    if Hnew > self.Hmin:
                        erZ = abs((Hnewt - Hnew) / Hnew)
                    else:
                        erZ = 0.0

                elif MDQZ == 2:  # Absolute value
                    erq = abs(Qnew_itr[_] - Qnew[_])
                    erZ = abs(Znew_itr[_] - Znew[_])
                else:
                    raise Exception('Code err')

                if erq > errQ:
                    errQ = erq
                if erZ > errZ:
                    errZ = erZ

                # Ensure THE CHANNEL IS INUNDATED
                if Znew[_] <= self.CSZBmn[_]:
                    Znew[_] = self.CSZBmn[_] + self.DHmin

            if errQ <= self.EPSQD and errZ <= self.EPSZW:
                break

            if itr == self.ITSUMF - 1:
                break

        # Process the result
        Qend = copy.deepcopy(Qnew)
        Zend = copy.deepcopy(Znew)

        # Ensure minimum water levels and flow rates
        for _ in range(self.Imax):
            if Zend[_] < self.CSZBmn[_]:  # Ensure water presence
                Zend[_] = self.CSZBmn[_] + self.DHmin
                Qend[_] = self.Qmin
            if Qend[_] < self.Qmin:
                Qend[_] = self.Qmin
                Zend[_] = self.CSZBmn[_] + self.DHmin

        return Qend, Zend

    def comp_qzba2(self, Zold, Znew):

        # Retrieve the A B DMK ALF at current itr
        AACS, BBCS, DMKCS, ALFCS = [], [], [], []
        Zmid, DADXCS = [], []

        for _ in range(self.Imax):
            ZWL = Znew[_]
            NMC = self.NMC1CS[_]  # Number of main channels in the section
            AAi, BBi, DMKi, alfi = compute_channel_char(self.Hmin, self.DHmin, self.DBIJ[_], self.ZBIJ[_], self.KNIJ[_], self.RNIJ[_], self.CSZW[_])
            AACS.append(AAi)
            BBCS.append(BBi)
            DMKCS.append(DMKi)
            ALFCS.append(alfi)

        for _ in range(self.Imax - 1):
            Zmid.append(0.5 * self.CitaFW * (Znew[_ + 1] + Znew[_]) + 0.5 * (1.0 - self.CitaFW) * (Zold[_ + 1] + Zold[_]))

        for _ in range(self.Imax - 1):
            Zmd = Zmid[_]
            AACS1 = compute_channel_char(self.Hmin, self.DHmin, self.DBIJ[_], self.ZBIJ[_], self.KNIJ[_], self.RNIJ[_], Zmd, simplified_factor=True)
            AACS2 = compute_channel_char(self.Hmin, self.DHmin, self.DBIJ[_ + 1], self.ZBIJ[_ + 1], self.KNIJ[_ + 1], self.RNIJ[_ + 1], Zmd, simplified_factor=True)
            DADXCS.append((AACS2 - AACS1) / self.DX2CS[_])

        return AACS, BBCS, DMKCS, ALFCS, DADXCS


if __name__ == "__main__":
    model = HydrodynamicModel_1D('G:\\A_1Dflow_sed\\Hydrodynamic_model\\')
    model.import_para('G:\\A_1Dflow_sed\\Hydrodynamic_model\\para\\MYR_GLBpara.dat',
                      'G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Original_cross_section\\cross_section_csv\\CSProf.csv',
                      'G:\\A_1Dflow_sed\\Hydrodynamic_model\\para\\MYR_FlwBound.csv',
                      'G:\\A_1Dflow_sed\\Hydrodynamic_model\\para\\MYR_QNRelt.csv')
    model._init_flow_boundary()
