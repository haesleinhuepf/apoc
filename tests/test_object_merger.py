def test_object_merger():
    import apoc
    from skimage.io import imread, imshow
    import pyclesperanto_prototype as cle
    import numpy as np

    image = imread('demo/membrane2d.tif')

    background_subtracted = cle.divide_by_gaussian_background(image, sigma_x=10, sigma_y=10)

    oversegmented = imread("demo/membrane2d_oversegmented.tif")

    annotation = imread("demo/membrane2d_merge_annotation.tif")

    feature_definition = "touch_portion mean_touch_intensity"

    classifier_filename = "object_merger.cl"

    apoc.erase_classifier(classifier_filename)
    classifier = apoc.ObjectMerger(opencl_filename=classifier_filename)

    classifier.train(features=feature_definition,
                     labels=oversegmented,
                     sparse_annotation=annotation,
                     image=background_subtracted)

    merged_labels = classifier.predict(labels=oversegmented, image=background_subtracted)

    assert merged_labels.max() == 31

def test_features():
    import apoc
    from skimage.io import imread, imshow
    import pyclesperanto_prototype as cle
    import numpy as np

    image = imread('demo/membrane2d.tif')

    background_subtracted = cle.divide_by_gaussian_background(image, sigma_x=10, sigma_y=10)

    oversegmented = imread("demo/membrane2d_oversegmented.tif")

    annotation = imread("demo/membrane2d_merge_annotation.tif")

    feature_definition = " ".join(['mean_touch_intensity',
            'touch_portion',
            'touch_count',
            'centroid_distance',
            'mean_intensity_difference',
            'standard_deviation_intensity_difference',
            'area_difference',
            'mean_max_distance_to_centroid_ratio_difference'])

    classifier_filename = "object_merger.cl"

    apoc.erase_classifier(classifier_filename)
    classifier = apoc.ObjectMerger(opencl_filename=classifier_filename)


    classifier.train(features=feature_definition,
                     labels=oversegmented,
                     sparse_annotation=annotation,
                     image=background_subtracted)
