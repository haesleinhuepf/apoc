from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd

from ._pixel_classifier import PixelClassifier


class TableRowClassifier:
    def __init__(
            self,
            opencl_filename="temp_table_row_classifier.cl",
            max_depth: int = 2, num_ensembles: int = 10,
            overwrite_classname: Optional[str] = None,
    ):
        """A RandomForestClassifier for classifying rows of a table that converts itself to OpenCL after training.

        Parameters
        ----------
        opencl_filename : str (optional)
            The path to which the openCL classifier will be saved.
        max_depth : int (optional)
            The maximum depth of the tree.
        num_ensembles : int (optional)
            The number of trees in the random forest.
        See Also
        --------
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        self._ordered_feature_names = []

        if overwrite_classname is None:
            overwrite_classname = self.__class__.__name__
        self._classifier_classname = overwrite_classname
        self.classifier = PixelClassifier(
            opencl_filename=opencl_filename,
            max_depth=max_depth,
            num_ensembles=num_ensembles,
            overwrite_classname=self.classifier_classname
        )

        # in case file existed, read feature specification from classifier
        from os.path import exists
        if exists(opencl_filename):
            self.feature_specification = self.classifier.feature_specification
            self._ordered_feature_names = self.feature_specification.replace(",", " ").split(" ")

    @property
    def ordered_feature_names(self) -> List[str]:
        """The feature names used in the order they are used by the classifier.

        This is set by self._prepare_feature_table()

        Returns
        -------
        ordered_feature_names : List[str]
            A list of the feature names in the order they are expected by the classifer.
        """
        return self._ordered_feature_names

    @property
    def classifier_classname(self) -> str:
        """The name used for the classifier class.

        This is set in the __init__ by the overwrite_classname kwarg.

        Returns
        -------
        classifier_classname : str
            The name used for the classifier class.
        """
        return self._classifier_classname

    def train(
        self,
        feature_table: Dict[str, Union[List[float], np.ndarray]],
        ground_truth: np.ndarray,
        continue_training: bool = False
    ):
        """Train a classifier that can differentiate classes from rows of pre-calculated features.

        Parameters
        ----------
        feature_table : Dict[str, Union[List[float], np.ndarray]]
            The table from which to make the prediction. Each row of the table
            will be classified. The table can either be a pandas DataFrame or a
            Dict with string keys (column names) and numpy array columns.
        ground_truth : np.array
            The array containing the ground truth class for each row in feature_table
        continue_training : bool
            Flag set to true if training is to be continued from an existing classifier.
            The default value is False.

        """
        ordered_features, ground_truth = self._prepare_feature_table(feature_table, ground_truth)
        self.classifier.train(ordered_features, ground_truth, continue_training=continue_training)
        self.classifier.to_opencl_file(self.classifier.opencl_file, overwrite_classname=self.classifier_classname)
        self.feature_specification = self.classifier.feature_specification

    def predict(
            self,
            feature_table: Dict[str, Union[List[float], np.ndarray]],
            return_numpy: bool = True
    ) -> np.array:
        """Predict row class from a table.

        Parameters
        ----------
        feature_table : Dict[str, Union[List[float], np.ndarray]]
            The table from which to make the prediction. Each row of the table
            will be classified. The table can either be a pandas DataFrame or a
            Dict with string keys (column names) and numpy array columns.
        return_numpy : bool
            If True, the resulting predictions are returned as a numpy array.
            If False, the predictions are returned as a pyopencl array.
            The default value is True.

        Returns
        -------
        output : np.ndarray
            An array containing the predicted class for each row.
        """
        import pyclesperanto_prototype as cle

        ordered_features = self.order_feature_table(feature_table)

        # allocate the result
        if len(ordered_features[0].shape) > 1:
            output = cle.create_like(ordered_features[0].shape)
        else:
            # make sure it's at least 2D
            output = cle.create((1, len(ordered_features[0])))

        # push the features
        parameters = {}
        for i, f in enumerate(ordered_features):
            parameters['in' + str(i)] = cle.push(f)
        parameters['out'] = output

        # run the classifier
        cle.execute(None, self.classifier.opencl_file, "predict", ordered_features[0].shape, parameters)

        # determine rows which contained NaN values
        table = pd.DataFrame(ordered_features)
        was_not_nan = cle.asarray([1 - np.max(table.isnull().values, axis=0) * 1])

        # mask output: if there was NaN before, classification becomes 0
        output = was_not_nan * output

        if return_numpy is True:
            return np.asarray(output[0], dtype=np.uint32)
        else:
            return output[0]

    def _prepare_feature_table(
            self,
            feature_table: Dict[str, Union[List[float], np.ndarray]],
            ground_truth : np.ndarray
    ) -> List[np.ndarray]:
        """Prepare a feature table for training.

        This coerces the feature table into the form expected by the classifier
        (list of numpy array) and stores the order of the features.

        Parameters
        ----------
        feature_table : Dict[str, Union[List[float], np.ndarray]]
            The table from which to make the prediction. Each row of the table
            will be classified. The table can either be a pandas DataFrame or a
            Dict with string keys (column names) and numpy array columns.
        ground_truth : np.array
            The array containing the ground truth class for each row in feature_table

        Returns
        -------
        ordered_features : List[np.ndarray]
            The features stored in a list. The order of the features is
            specified by self.ordered_feature_names
        """
        # make sure it's a DataFrame and we can modify it
        feature_table = pd.DataFrame(feature_table).copy()

        # store original keys
        original_keys = feature_table.keys().copy()

        # add ground_truth to table so that we can filter it with the other columns
        if "ground_truth" in feature_table.keys():
            raise ValueError("feature_table must not contain column named 'ground_truth'")
        feature_table['ground_truth'] = ground_truth

        # drop rows with NaN values
        feature_table = feature_table.dropna(how="any")

        # drop rows with ground_truth == 0
        feature_table = feature_table[feature_table['ground_truth'] > 0]

        # get ground_truth back and a table with the original columns
        updated_ground_truth = np.asarray(feature_table['ground_truth'])
        feature_table = feature_table[original_keys]

        self._ordered_feature_names = list(feature_table.keys())
        self.classifier.feature_specification = " ".join(self.ordered_feature_names)
        return self.order_feature_table(feature_table), updated_ground_truth

    def order_feature_table(self, feature_table: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Coerce a feature table into the format required by the classifier.

        Parameters
        ----------
        feature_table : Dict[str, Union[List[float], np.ndarray]]
            The table from which to make the prediction. Each row of the table
            will be classified. The table can either be a pandas DataFrame or a
            Dict with string keys (column names) and numpy array columns.

        Returns
        -------
        ordered_features : List[np.ndarray]
            The features stored in a list. The order of the features is
            specified by self.ordered_feature_names
        """
        return [np.asarray(feature_table[feature]) for feature in self.ordered_feature_names]

    def statistics(self):
        """Provide statistics about the trained Random Forest Classifier

        After training or loading a model, this function reads out the decision trees and generates
        statistics from it. It counts for each decision depth how often given features are taken into
        account. It returns two dictionaries. Both dictionaries contain the feature names used during
        training as keys. The values are lists with numbers: The first 'ratios' dictionary contains
        numbers between 0 and 1. The higher that number, the more often is the given feature taken
        into account on the given decision level. The second 'count' dictionary contains the count of
        decisions taking a given feature into account. For example in case the result looks like this:

        A: [0.3, 0.1], [30, 20]
        B: [0.7, 0.9], [70, 180]

        Feature A was taken into account 30% of the decision trees on the first level and in 10% on
        the second level. In case of 100 trees, these are 30 trees on the first level and can be up
        to 20 trees on the second level. Each level doubles the number of available decisions in these
        binary decision trees.

        Returns
        -------
        ratios: dict
        counts: dict
        """
        return self.classifier.statistics()
