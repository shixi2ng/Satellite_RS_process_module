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
        self.NDVI = 'NDVI = (B8 - B4) / (B8 + B4)'
        self.NDVI_20m = 'NDVI_20m = (B8A - B4) / (B8A + B4)'
        self.OSAVI_20m = 'OSAVI_20m = 1.16 * (B8A - B4) / (B8A + B4 + 0.16)'
        self.OSAVI = 'OSAVI = 1.16 * (B8 - B4) / (B8 + B4 + 0.16)'
        self.AWEI = 'AWEI = 4 * (B3 - B11) - (0.25 * B8 + 2.75 * B12)'
        self.AWEInsh = 'AWEInsh = B2 + 2.5 * B3 - 0.25 * B12 - 1.5 * (B8 + B11)'
        self.MNDWI = 'MNDWI = (B3 - B11) / (B3 + B11)'
        self.EVI = 'EVI = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1)'
        self.EVI2 = 'EVI2 = 2.5 * (B8 - B4) / (B8 + 2.4 * B4 + 1)'
        self.GNDVI = 'GNDVI = (B8 - B3) / (B8 + B3)'
        self.NDVI_RE = 'NDVI_RE = (B7 - B5) / (B7 + B5)'
        self.NDVI_RE2 = 'NDVI_RE2 = (B8 - B5) / (B8 + B5)'
        self.IEI = 'IEI = 1.5 * (B8 - B4) / (B8 + B4 + 0.5)'

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

