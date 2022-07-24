import numpy as np


def test_object_classification():
    image      = np.asarray([[0,0,1,1,2,2,3,3,3,3,3]])
    labels     = np.asarray([[0,0,1,1,1,1,2,2,2,2,2]])
    annotation = np.asarray([[0,0,2,2,2,0,0,0,0,1,1]])
    reference  = np.asarray([[0,0,1,1,1,1,1,1,1,1,1]])
    feature_definition = """
            area
            min_intensity max_intensity sum_intensity mean_intensity standard_deviation_intensity
            mass_center_x mass_center_y mass_center_z
            centroid_x centroid_y centroid_z
            max_distance_to_centroid max_distance_to_mass_center
            mean_max_distance_to_centroid_ratio mean_max_distance_to_mass_center_ratio
            touching_neighbor_count average_distance_of_touching_neighbors
            """.replace("\n", " ")

    import apoc
    filename = "test_object_classification.cl"
    apoc.erase_classifier(filename)
    oc = apoc.ObjectClassifier(opencl_filename=filename, num_ensembles=10)

    assert len(oc.feature_importances().keys()) == 0

    print(oc)
    assert 'ObjectClassifier' in str(oc)

    oc.train(feature_definition, labels, annotation, image)

    print(oc.feature_importances().keys())

    assert len(oc.feature_importances().keys()) == 18

    print(oc)
    assert 'ObjectClassifier' in str(oc)

    result = oc.predict(labels, image)

    assert result.dtype == np.uint32

    print(result)

    assert np.allclose(reference, result)

    feature_importances = oc.feature_importances()
    assert feature_importances["area"] < 0.1
    assert feature_importances["min_intensity"] < 0.1
    assert feature_importances["max_intensity"] > 0.1
    assert feature_importances["sum_intensity"] > 0.1
    assert feature_importances["sum_intensity"] > 0.1
    assert feature_importances["standard_deviation_intensity"] > 0.1
    assert feature_importances["mass_center_x"] < 0.1
    assert feature_importances["mass_center_y"] < 0.1
    assert feature_importances["mass_center_z"] < 0.1
    assert feature_importances["centroid_x"] < 0.1
    assert feature_importances["centroid_y"] < 0.1
    assert feature_importances["centroid_z"] < 0.1
    assert feature_importances["max_distance_to_centroid"] > 0.1
    assert feature_importances["max_distance_to_mass_center"] > 0.1
    assert feature_importances["mean_max_distance_to_centroid_ratio"] < 0.1
    assert feature_importances["mean_max_distance_to_mass_center_ratio"] < 0.1
    assert feature_importances["touching_neighbor_count"] < 0.1
    assert feature_importances["average_distance_of_touching_neighbors"] < 0.1

    print(oc)
    assert 'ObjectClassifier' in str(oc)


def test_illegal_feature_name_causes_error():
    def test_object_classification():
        image = np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]])
        labels = np.asarray([[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]])
        annotation = np.asarray([[0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 1]])
        reference = np.asarray([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        feature_definition = """
                area
                average_distance_of_n_nearest_neighbors
                """.replace("\n", " ")

        import apoc
        oc = apoc.ObjectClassifier(num_ensembles=10)

        # this should cause an exception
        #import pytest
        #with pytest.raises(Exception):
        try:
            oc.train(feature_definition, labels, annotation, image)
            assert "" == "This should have caused an exception"
        except:
            pass


def test_object_classification_statistics():
    image = np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]])
    labels = np.asarray([[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]])
    annotation = np.asarray([[0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 1]])
    reference = np.asarray([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    feature_definition = """
            area mean_intensity standard_deviation_intensity touching_neighbor_count
            """.replace("\n", " ")

    import apoc
    oc = apoc.ObjectClassifier()
    oc.train(feature_definition, labels, annotation, image)

    shares, _ = oc.statistics()
    assert len(shares.keys()) == 4
