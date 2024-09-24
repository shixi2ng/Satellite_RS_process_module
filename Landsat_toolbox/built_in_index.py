# coding=utf-8
import sympy


def convert_index_func(expr: str):
    try:
        f = sympy.sympify(expr)
        dep_list = sorted(f.free_symbols, key=str)
        num_f = sympy.lambdify(dep_list, f)
        return dep_list, num_f
    except:
        raise ValueError(f'The {expr} is not valid!')


class built_in_index(object):

    def __init__(self, *args):
        self.NDVI = 'NDVI = (NIR - RED) / (NIR + RED)'
        self.OSAVI = 'OSAVI = 1.16 * (NIR - RED) / (NIR + RED + 0.16)'
        self.AWEI = 'AWEI = 4 * (GREEN - SWIR) - (0.25 * NIR + 2.75 * SWIR2)'
        self.AWEInsh = 'AWEInsh = BLUE + 2.5 * GREEN - 0.25 * SWIR2 - 1.5 * (NIR + SWIR)'
        self.MNDWI = 'MNDWI = (GREEN - SWIR) / (SWIR + GREEN)'
        self.EVI = 'EVI = 2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)'
        self.EVI2 = 'EVI2 = 2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)'
        self.SVVI = "SVVI = (sqrt(((BLUE - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2 + (GREEN - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2 + (RED - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2 + (NIR - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2 + (SWIR - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2 + (SWIR2 - ((BLUE + GREEN + RED + NIR + SWIR + SWIR2) / 6)) ** 2) / 6) - sqrt(((NIR - ((NIR + SWIR + SWIR2) / 3)) ** 2 + (SWIR - ((NIR + SWIR + SWIR2) / 3)) ** 2 + (SWIR2 - ((NIR + SWIR + SWIR2) / 3)) ** 2) / 3)) * 0.000275"
        self.TCGREENESS = 'TCGREENESS = (- 0.1603 * BLUE - 0.2819 * GREEN - 0.4934 * RED + 0.7940 * NIR - 0.0002 * SWIR - 0.1446 * SWIR2) / 10000'
        self.BLUE = 'BLUE = BLUE'
        self.GREEN = 'GREEN = GREEN'
        self.RED = 'RED = RED'
        self.NIR = 'NIR = NIR'
        self.SWIR = 'SWIR = SWIR'
        self.SWIR2 = 'SWIR2 = SWIR2'

        self._exprs2index(*args)
        self._built_in_index_dic()

    def _exprs2index(self, *args):
        for temp in args:
            if type(temp) is not str:
                raise ValueError(f'{temp} expression should be in a str type!')
            elif '=' in temp:
                self.__dict__[temp.split('=')[0]] = temp
            else:
                raise ValueError(f'{temp} expression should be in a str type!')

    def add_index(self, *args):
        self._exprs2index(*args)
        self._built_in_index_dic()

    def _built_in_index_dic(self):
        self.index_dic = {}
        for i in self.__dict__:
            if i != 'index_dic':
                var, func = convert_index_func(self.__dict__[i].split('=')[-1])
                self.index_dic[i] = [var, func]


