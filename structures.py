from collections import defaultdict

class DotDict(dict):
    """dot.notation access to dictionary attributes
    Using it like a struct to group some variables together.
    I find it easier to make copies and edit params this way."""

    def __getattr__(self, item):
        if item in self: # fail silently
            return self[item]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

class SparseList2D:
    """2D sparse list with O(1) insert and O(log N) lower_bound query
    first dimension stores sparse individual elements
    second dimension stores sparse prefix sum for numerical data"""

    def __init__(self):
        self._index = {}
        self._data = {}
        self._len = {}

    def sum(self, element, index1, index2):
        """remember to call calc_lengths first
        returns the sum of data between 2 indexes in an element"""
        first = self.lower_bound(element, index1)
        if first:
            first = self._data[element][first - 1]
        second = self.lower_bound(element, index2)
        if second:
            second = self._data[element][second - 1]
        return second - first

    def lower_bound(self, element, index):
        """remember to call calc_lengths first
        returns the number of set indexes before or at index"""
        if element not in self._index:
            return 0
        if index < self._index[element][0]:
            return 0
        l = 0
        r = self._len[element]
        while l < r - 1:
            n = (l + r) // 2
            if self._index[element][n] <= index:
                l = n
            else:
                r = n - 1
        return l + 1

    def set(self, element, index, data):
        """List[element][index] = data
        indexes have to be set in ascending order!!"""
        if element in self._index:
            self._index[element].append(index)
            self._data[element].append(self._data[element][-1] + data)
        else:
            self._index[element] = [index]
            self._data[element] = [data]

    def calc_lengths(self):
        """pre-calculate lengths for faster query
        call once after data creation and forget"""
        for i in self._data:
            self._len[i] = len(self._data[i]) - 1

    def first(self, element):
        """returns the first set item of a 1D array within the 2D matrix"""
        if element in self._index:
            return self._data[element][0]

    def before(self, element, index):
        """returns the first value before the index"""
        if element not in self._index:
            return None
        if index < self._index[element][0]:
            return None
        l = 0
        r = self._len[element]
        while l < r - 1:
            n = (l + r) // 2
            if self._index[element][n] < index:
                l = n
            else:
                r = n - 1
        return self._data[element][l]

def Tree():
    """very basic tree"""

    return defaultdict(Tree)
