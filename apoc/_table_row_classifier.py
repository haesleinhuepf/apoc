from typing import Dict, List, Union

import numpy as np

from ._pixel_classifier import PixelClassifier


class TableRowClassifier():
    def __init__(self, opencl_filename="temp_object_classifier.cl", max_depth: int = 2, num_ensembles: int = 10):
        """
        A RandomForestClassifier for label classification that converts itself to OpenCL after training.

        Parameters
        ----------
        opencl_filename : str (optional)
        max_depth : int (optional)
        num_ensembles : int (optional)

        See Also
        --------
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        self._ordered_feature_names = []

        self.classifier = PixelClassifier(
            opencl_filename=opencl_filename,
            max_depth=max_depth,
            num_ensembles=num_ensembles,
            overwrite_classname=self.__class__.__name__
        )

    @property
    def ordered_feature_names(self) -> List[str]:
        """
        Returns
        -------
        ordered_feature_names : List[str]
            A list of the feature names in the order they are expected by the classifer.
        """
        return self._ordered_feature_names

    def train(self, feature_table: Dict[str, Union[List[float], np.ndarray]], gt: np.ndarray, continue_training: bool = False):
        """
        Train a classifier that can differentiate label types according to intensity, size and shape.

        Parameters
        ----------
        features: Space separated string containing those:
            'area',
            'min_intensity', 'max_intensity', 'sum_intensity', 'mean_intensity', 'standard_deviation_intensity',
            'mass_center_x', 'mass_center_y', 'mass_center_z',
            'centroid_x', 'centroid_y', 'centroid_z',
            'max_distance_to_centroid', 'max_distance_to_mass_center',
            'mean_max_distance_to_centroid_ratio', 'mean_max_distance_to_mass_center_ratio',
            'touching_neighbor_count', 'average_distance_of_touching_neighbors', 'average_distance_of_n_nearest_neighbors'
        labels: label image
        sparse_annotation: label image with annotations. If one label is annotated with multiple classes, the
            maximum is considered while training.
        image: intensity image (optional)

        """
        ordered_features = self._prepare_feature_table(feature_table)
        self.classifier.train(ordered_features, gt, continue_training=continue_training)
        self.classifier.to_opencl_file(self.classifier.opencl_file, overwrite_classname=self.__class__.__name__)

    def predict(
            self,
            feature_table: Dict[str, Union[List[float], np.ndarray]],
            return_numpy: bool = True
    ) -> List[int]:
        """Predict object class from label image and optional intensity image.

        Parameters
        ----------
        labels: label image
        image: intensity image

        Returns
        -------
        label image representing a semantic segmentation: pixel intensities represent label class

        """
        import pyclesperanto_prototype as cle

        ordered_features = self.order_feature_table(feature_table)

        # allocate the result
        output = cle.create_like(ordered_features[0].shape)

        # push the features
        parameters = {}
        for i, f in enumerate(ordered_features):
            parameters['in' + str(i)] = cle.push(f)
        parameters['out'] = output

        # run the classifier
        cle.execute(None, self.classifier.opencl_file, "predict", ordered_features[0].shape, parameters)

        if return_numpy is True:
            return np.asarray(output, dtype=np.uint32)
        else:
            return output

    def _prepare_feature_table(self, feature_table) -> List[np.ndarray]:
        if len(self._ordered_feature_names) == 0:
            # if the feature names haven't previously been stored,
            # store them as a list
            self._ordered_feature_names = list(feature_table.keys())
        ordered_feature_names = self._ordered_feature_names
        return [
            np.asarray(feature_table[feature_name]) for feature_name in ordered_feature_names
        ]

    def order_feature_table(self, feature_table: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return [np.asarray(feature_table[feature]) for feature in self.ordered_feature_names]

