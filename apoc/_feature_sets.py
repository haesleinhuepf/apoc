from enum import Enum

def _make_feature_definition(filters, radii):
    """Turns a list of filter names and radii into a string that works as valid feature definition in APOC.
    E.g. a list of filers ['A', 'B'] and radii [1, 2] will result in:
    "A=1 A=2 B=1 B=2"

    Parameters
    ----------
    filters: list[str]
    radii: list[int or float]

    Returns
    -------
    str
    """
    result = ""
    for f in filters:
        for r in radii:
            result = result + " " + f + "=" + str(r)
    return result.strip()

class PredefinedFeatureSet(Enum):
    """
    Predefined feature sets make it easy to select features in a pulldown.

    See Also
    --------
    .. https://github.com/haesleinhuepf/apoc/blob/main/demo/feature_stacks.ipynb
    """
    custom = ""
    small_quick = "original " + _make_feature_definition(["gaussian_blur", "sobel_of_gaussian_blur"], [1])
    medium_quick = _make_feature_definition(["gaussian_blur", "sobel_of_gaussian_blur"], [5])
    large_quick = _make_feature_definition(["gaussian_blur", "sobel_of_gaussian_blur"], [25])

    small_dog_log = "original " + _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], [1])
    medium_dog_log = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], [5])
    large_dog_log = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], [25])

    object_size_1_to_2_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(1, 3))
    object_size_1_to_5_px = "original " + _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(1, 6))
    object_size_3_to_8_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(3, 9))
    object_size_5_to_10_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(5, 11))
    object_size_10_to_15_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(10, 16))
    object_size_15_to_20_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(15, 21))
    object_size_20_to_25_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(20, 26))
    object_size_25_to_50_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(25, 50, 5))
    object_size_50_to_100_px = _make_feature_definition(["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"], range(50, 100, 10))
