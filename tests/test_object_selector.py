def test_object_selector():
    from skimage.io import imread
    import pyclesperanto_prototype as cle
    import pandas as pd
    import numpy as np
    import apoc

    cle.select_device('RTX')

    image = imread('demo/blobs.tif')
    labels = imread('demo/labels.tif')
    annotation = imread('demo/label_annotation.tif')

    features = 'area,mean_max_distance_to_centroid_ratio,standard_deviation_intensity'

    cl_filename = "object_selector.cl"

    # Create an object classifier
    apoc.erase_classifier(cl_filename)  # delete it if it was existing before
    classifier = apoc.ObjectSelector(cl_filename, positive_class_identifier=1)

    # train it
    classifier.train(features, labels, annotation, image)

    # determine object classification
    result = classifier.predict(labels, image)

    assert result.max() == 23

    # now, we reload the classifier from disc:
    classifier = apoc.ObjectSelector(cl_filename)

    result = classifier.predict(labels.T, image.T)

    assert result.max() == 23

    fi = classifier.feature_importances()
    assert abs(fi['area'] - 0.3) < 0.1
    assert abs(fi['mean_max_distance_to_centroid_ratio'] - 0.4) < 0.1
    assert abs(fi['standard_deviation_intensity'] - 0.3) < 0.1
