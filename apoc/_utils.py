from typing import Union

import pyclesperanto_prototype as cle
import inspect
import pyopencl
from ._feature_sets import PredefinedFeatureSet

def generate_feature_stack(image, features_specification : Union[str, PredefinedFeatureSet] = None):
    """
    Creates a feature stack from a given image.

    Parameters
    ----------
    image : ndarray
        2D or 3D image to generate a feature stack from
    features_specification : str or PredefinedFeatureSet
        a space-separated list of features, e.g.
        original gaussian=4 sobel_of_gaussian=4 or a
        PredefinedFeatureSet

    Returns
    -------
    a list of OCLarray images
    """

    image = cle.push(image)

    # default features
    if features_specification is None:
        blurred = cle.gaussian_blur(image, sigma_x=2, sigma_y=2, sigma_z=2)
        edges = cle.sobel(blurred)
        stack = [
            image,
            blurred,
            edges
        ]

        return stack
    if isinstance(features_specification, PredefinedFeatureSet):
        features_specification = features_specification.value

    while "  " in features_specification:
        features_specification = features_specification.replace("  ", " ")
    while "\t" in features_specification:
        features_specification = features_specification.replace("\t", " ")

    features_specs = features_specification.split(" ")
    generated_features = {}

    result_features = []

    for spec in features_specs:
        if spec.lower() == 'original':
            generated_features['original'] = image
            result_features.append(image)
        elif "=" in spec:
            temp = spec.split("=")
            operation = temp[0]
            numeric_parameter = float(temp[1])

            if not hasattr(cle, operation) and "_of_" in operation:
                temp = operation.split("_of_")
                outer_operation = temp[0]
                inner_operation = temp[1]

                if (inner_operation+"="+str(numeric_parameter)) not in generated_features.keys():
                    new_image = cle.create_like(image)
                    _apply_operation(inner_operation, image, new_image, numeric_parameter)
                    generated_features[inner_operation+"="+str(numeric_parameter)] = new_image

                if (operation+"="+str(numeric_parameter)) not in generated_features.keys():
                    new_image2 = cle.create_like(image)
                    _apply_operation(outer_operation, generated_features[inner_operation+"="+str(numeric_parameter)], new_image2, numeric_parameter)
                    generated_features[operation+"="+str(numeric_parameter)] = new_image2
            else:
                if (operation+"="+str(numeric_parameter)) not in generated_features:
                    new_image = cle.create_like(image)
                    _apply_operation(operation, image, new_image, numeric_parameter)
                    generated_features[operation+"="+str(numeric_parameter)] = new_image

            result_features.append(generated_features[operation+"="+str(numeric_parameter)])

    return result_features

def _apply_operation(operation, image, new_image, numeric_parameter):
    func = getattr(cle, operation)
    sig = inspect.signature(func)
    if len(sig.parameters.keys()) == 2:
        func(image, new_image)
    elif len(sig.parameters.keys()) == 3:
        func(image, new_image, numeric_parameter)
    elif len(sig.parameters.keys()) == 4:
        func(image, new_image, numeric_parameter, numeric_parameter)
    elif len(sig.parameters.keys()) == 5:
        func(image, new_image, numeric_parameter, numeric_parameter, numeric_parameter)
    elif len(sig.parameters.keys()) == 8:
        # e.g. difference_of_gaussian
        func(image, new_image, numeric_parameter * 0.9, numeric_parameter * 0.9, numeric_parameter * 0.9, numeric_parameter * 1.1, numeric_parameter * 1.1, numeric_parameter * 1.1)
    else:
        func(image, new_image, numeric_parameter, numeric_parameter, numeric_parameter)
