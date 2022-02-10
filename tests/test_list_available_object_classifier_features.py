def test_list_available_object_classifier_features():
    from apoc import list_available_object_classification_features

    assert len(list_available_object_classification_features()) == 41