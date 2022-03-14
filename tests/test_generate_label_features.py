def test_label_feature_generation():
    import numpy as np
    import apoc

    image = np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]])
    labels = np.asarray([[0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 2]])
    annotation = np.asarray([[0, 0, 2, 2, 2, 0, 3, 0, 0, 1, 1]])

    feature_definition = """
            area
            """.replace("\n", " ")

    oc = apoc.ObjectClassifier()
    table, ground_truth = oc._make_features(feature_definition, labels, annotation, image)

    # there are 3 labels
    assert len(ground_truth) == 3

    # we only measured area, 1 feature
    assert len(table) == 1

    # there are three area measurements
    assert len(table[0][0]) == 3


def test_label_feature_generation_with_annotated_background():
    import numpy as np
    import apoc

    image = np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]])
    labels = np.asarray([[0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 2]])
    annotation = np.asarray([[1, 0, 2, 2, 2, 0, 3, 0, 0, 1, 1]])

    feature_definition = """
            area
            """.replace("\n", " ")

    oc = apoc.ObjectClassifier()
    table, ground_truth = oc._make_features(feature_definition, labels, annotation, image)

    # there are 3 labels
    assert len(ground_truth) == 3

    # we only measured area, 1 feature
    assert len(table) == 1

    # there are three area measurements
    assert len(table[0][0]) == 3


def test_label_feature_generation_for_prediction():
    import numpy as np
    import apoc

    image = np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3]])
    labels = np.asarray([[0, 0, 1, 1, 1, 1, 3, 2, 2, 2, 2]])

    feature_definition = """
            area
            """.replace("\n", " ")

    oc = apoc.ObjectClassifier()
    table, _ = oc._make_features(feature_definition, labels, None, image)

    # we only measured area, 1 feature
    assert len(table) == 1

    # there are four area measurements: background + 3 labels
    assert len(table[0][0]) == 4
