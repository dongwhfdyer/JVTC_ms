import warnings
from typing import Union
import mindspore
import numpy as np


# get the statistics of numpy array
def get_statistics(numpy_array: Union[np.ndarray, mindspore.Tensor], name=None):
    """
    get the statistics of numpy array
    :param numpy_array: numpy array
    :return: none
    It will print many statistics of numpy array.
    """
    # if it's tensor, then convert to numpy array
    print()
    if name is not None:
        print("################################################## %s" % name)
    else:
        print("##################################################")
    if isinstance(numpy_array, mindspore.Tensor):
        try:
            numpy_array = numpy_array.numpy()
        except:
            warnings.warn("Can't convert to numpy array. Maybe data is on GPU.", UserWarning)
            numpy_array = numpy_array.cpu().detach().numpy()
    statistics_dict = {
        "mean": np.mean(numpy_array),
        "std": np.std(numpy_array),
        "max": np.max(numpy_array),
        "min": np.min(numpy_array),
        "median": np.median(numpy_array),
        "variance": np.var(numpy_array)
    }
    for key in statistics_dict:
        print("%s: %.4f" % (key, statistics_dict[key]))
    print("##################################################")
    print()
    return statistics_dict
