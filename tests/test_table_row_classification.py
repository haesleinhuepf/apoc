import numpy as np
import pandas as pd

feature_table_dict = {
    "area": np.array([10, 10, 30, 30, 10]),
    "surface_area": np.array([5, 3, 80, 80, 9])
}
feature_table_data_frame = pd.DataFrame(feature_table_dict)
ground_truth = np.array([1, 1, 2, 2, 1])


def test_table_row_classification_from_dict():
    """Test the TableRowClassifier when the train/predict data are
    a dictionary with keys for columns names and columns stored
    as numpy arrays.
    """

    import apoc
    oc = apoc.TableRowClassifier()
    oc.train(
        feature_table=feature_table_dict,
        gt=ground_truth,
        continue_training=False
    )
    result = oc.predict(feature_table_dict, return_numpy=True)

    assert result.dtype == np.uint32

    print(result)

    assert np.allclose(ground_truth, result)


def test_table_row_classification_from_dataframe():
    """Test the TableRowClassifier when the train/predict data
    are a pandas dataframe.
    """
    import apoc
    oc = apoc.TableRowClassifier()
    oc.train(
        feature_table=feature_table_data_frame,
        gt=ground_truth,
        continue_training=False
    )
    result = oc.predict(feature_table_dict, return_numpy=True)

    assert result.dtype == np.uint32

    print(result)

    assert np.allclose(ground_truth, result)
