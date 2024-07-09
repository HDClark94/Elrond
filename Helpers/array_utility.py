import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


def pad_shorter_array_with_0s(array1, array2):
    if len(array1) > len(array2):
        array2 = np.concatenate([array2, np.zeros(len(array1) - len(array2))])
    elif len(array2) > len(array1):
        array1 = np.concatenate([array1, np.zeros(len(array2) - len(array1))])
    return array1, array2


# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def shift(array_to_shift, n):
    if n >= 0:
        return np.concatenate((np.full(n, np.nan), array_to_shift[:-n]))
    else:
        return np.concatenate((array_to_shift[-n:], np.full(-n, np.nan)))


'''
Shifts 2d array along given axis.

array_to_shift : 2d array that is to be shifted
n : array will be shifted by n places
axis : shift along this axis (should be 0 or 1)
'''


def shift_2d(array_to_shift, n, axis):
    shifted_array = np.zeros_like(array_to_shift)
    if axis == 0:  # shift along x axis
        if n == 0:
            return array_to_shift
        if n > 0:
            shifted_array[:, :n] = 0
            shifted_array[:, n:] = array_to_shift[:, :-n]
        else:
            shifted_array[:, n:] = 0
            shifted_array[:, :n] = array_to_shift[:, -n:]

    if axis == 1:  # shift along y axis
        if n == 0:
            return array_to_shift
        elif n > 0:
            shifted_array[-n:, :] = 0
            shifted_array[:-n, :] = array_to_shift[n:, :]
        else:
            shifted_array[:-n, :] = 0
            shifted_array[-n:, :] = array_to_shift[:n, :]
    return shifted_array


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
         # linear interpolation of NaNs
         nans, x= nan_helper(y)
         y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def remove_nans_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2


def remove_nans_and_inf_from_both_arrays(array1, array2):
    not_nans_in_array1 = ~np.isnan(array1)
    not_nans_in_array2 = ~np.isnan(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]

    not_nans_in_array1 = ~np.isinf(array1)
    not_nans_in_array2 = ~np.isinf(array2)
    array1 = array1[not_nans_in_array1 & not_nans_in_array2]
    array2 = array2[not_nans_in_array1 & not_nans_in_array2]
    return array1, array2


def pandas_collumn_to_numpy_array(pandas_series):
    new_array = []
    for i in range(len(pandas_series)):
        element = pandas_series.iloc[i]

        if len(np.shape(element)) == 0:
            new_array.append(element)
        else:
            new_array.extend(element)

    return np.array(new_array)


def list_of_list_to_1d_numpy_array(list_of_lists):
    new_array = []
    for i in range(len(list_of_lists)):
        for j in range(len(list_of_lists[i])):
            new_array.append(list_of_lists[i][j])
    return np.array(new_array)

def pandas_collumn_to_2d_numpy_array(pandas_series):
    new_array = []
    for i in range(len(pandas_series)):
        element = pandas_series.iloc[i]

        if len(np.shape(element)) == 0:
            new_array.append([element])
        else:
            new_array.append(element)

    return np.array(new_array)


def bool_to_int_array(bool_array):
    int_array = np.array(bool_array, dtype=np.int64)
    return int_array


def add_zero_to_data_if_empty(data):
    for i in range(len(data)):
        if len(data[i]) == 0:
            data[i] = np.array([0])
    return data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    array_to_shift = np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
    n = -2
    axis = 1

    desired_result = np.array([[np.nan, np.nan, 1, 1], [np.nan, np.nan, 2, 9], [np.nan, np.nan, 3, 3], [np.nan, np.nan, 4, 4], [np.nan, np.nan, 5, 5], [np.nan, np.nan, 6, 6]])
    result = shift_2d(array_to_shift, n, axis)

    array_to_shift2 = np.array([[[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3]], [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]])


if __name__ == '__main__':
    main()

