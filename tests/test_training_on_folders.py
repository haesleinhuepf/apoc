def test_training_on_folders():
    import apoc
    import os
    from skimage.io import imread
    import pyclesperanto_prototype as cle
    import matplotlib.pyplot as plt

    image_folder = "demo/folder/images/"
    masks_folder = "demo/folder/masks/"

    # setup classifer and where it should be saved
    segmenter = apoc.ObjectSegmenter(opencl_filename="test_on_folder.cl")

    # setup feature set used for training
    features = apoc.PredefinedFeatureSet.object_size_1_to_5_px.value

    # train classifier on folders
    apoc.train_classifier_from_image_folders(
        segmenter,
        features,
        image=image_folder,
        ground_truth=masks_folder)

    # test if the file was saved
    segmenter = apoc.ObjectSegmenter(opencl_filename="test_on_folder.cl")

    # apply classifier
    file_list = os.listdir(image_folder)
    for i, filename in enumerate(file_list):

        image = imread(image_folder + filename)

        # apply classifier
        labels = segmenter.predict(image)

        # There are at least 4 objects in each image
        assert(labels.max() > 3)

if __name__ == '__main__':
    test_training_on_folders()
