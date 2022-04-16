import numpy as np
import pandas as pd
import pytest

feature_table_dict = {
    "area": np.array([10, 10, 30, 30, 10]),
    "surface_area": np.array([5.0, 3, 80, 80, 9])
}
feature_table_data_frame = pd.DataFrame(feature_table_dict)
ground_truth = np.array([1, 1, 2, 2, 1])


@pytest.mark.parametrize("feature_table", [feature_table_dict, feature_table_data_frame])
def test_table_row_classification(tmpdir, feature_table):
    """Test the TableRowClassifier when the train/predict data are
    a dictionary with keys for columns names and columns stored
    as numpy arrays.
    """

    import apoc
    opencl_file = tmpdir.join("test.cl")
    oc = apoc.TableRowClassifier(opencl_file)
    oc.train(
        feature_table=feature_table,
        ground_truth=ground_truth,
        continue_training=False
    )
    result = oc.predict(feature_table, return_numpy=True)

    assert result.dtype == np.uint32
    assert np.allclose(ground_truth, result)

    # rerun classifier from file
    oc = apoc.TableRowClassifier(opencl_file)
    result = oc.predict(feature_table, return_numpy=True)

    assert result.dtype == np.uint32
    assert np.allclose(ground_truth, result)

@pytest.mark.parametrize("feature_table", [feature_table_dict, feature_table_data_frame])
def test_with_nans(tmpdir, feature_table):

    feature_table['surface_area'][4] = np.nan

    print(feature_table)

    annotation = np.array([1, 1, 2, 2, 0])
    reference = np.array([1, 1, 2, 2, 0])

    import apoc
    opencl_file = tmpdir.join("test.cl")
    oc = apoc.TableRowClassifier(opencl_file)
    oc.train(
        feature_table=feature_table,
        ground_truth=annotation,
        continue_training=False
    )
    result = oc.predict(feature_table, return_numpy=True)

    np.allclose(result, reference)

@pytest.mark.parametrize("feature_table", [feature_table_dict, feature_table_data_frame])
def test_with_nans_and_missing_annotations(tmpdir, feature_table):

    feature_table['surface_area'][4] = np.nan

    print(feature_table)

    annotation = np.array([1, 0, 0, 2, 0])
    reference = np.array([1, 1, 2, 2, 0])

    import apoc
    opencl_file = tmpdir.join("test.cl")
    oc = apoc.TableRowClassifier(opencl_file)
    oc.train(
        feature_table=feature_table,
        ground_truth=annotation,
        continue_training=False
    )
    result = oc.predict(feature_table, return_numpy=True)

    print(reference)
    print(result)

    np.allclose(result, reference)