import numpy as np
import pandas as pd
import pytest

feature_table_dict = {
    "area": np.array([10, 10, 30, 30, 10]),
    "surface_area": np.array([5, 3, 80, 80, 9])
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
        gt=ground_truth,
        continue_training=False
    )
    result = oc.predict(feature_table, return_numpy=True)

    assert result.dtype == np.uint32
    assert np.allclose(ground_truth, result)
