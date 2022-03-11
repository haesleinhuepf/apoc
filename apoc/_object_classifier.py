from ._table_row_classifier import TableRowClassifier
import numpy as np


class ObjectClassifier():
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
        self.FEATURE_SPECIFICATION_KEY = "feature_specification = "

        self.classifier = TableRowClassifier(
            opencl_filename=opencl_filename,
            max_depth=max_depth,
            num_ensembles=num_ensembles,
            overwrite_classname=self.__class__.__name__
    )

    def train(self, features: str, labels, sparse_annotation, image=None, continue_training : bool = False):
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
        self.classifier.feature_specification = features.replace(",", " ")
        selected_features, gt = self._make_features(self.classifier.feature_specification, labels, sparse_annotation, image)

        self.classifier.train(selected_features, gt, continue_training=continue_training)

    def predict(self, labels, image=None):
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
        labels = cle.push(labels)

        selected_features, _ = self._make_features(self.classifier.feature_specification, labels, None, image)
        output = self.classifier.predict(selected_features, return_numpy=False)

        # set background to zero
        cle.set_column(output, 0, 0)

        result_labels = cle.create_labels_like(labels)
        cle.replace_intensities(labels, output, result_labels)

        return result_labels

    def _make_features(self, features: str, labels, annotation=None, image=None):
        """Determine requested features. If annotation is provided, also a ground-truth vector will be returned.

        Parameters
        ----------
        features: str
            see train() function for explanation
        labels: ndimage (int)
        annotation: ndimage(int), optional
            sparse annotation label image
        image: ndimage, optional
            intensity image for e.g. mean intensity calculation

        Returns
        -------
        table: dict of vectors
        gt: vector
        """

        import pyclesperanto_prototype as cle
        pixel_statistics = cle.statistics_of_background_and_labelled_pixels(image, labels)

        if annotation is not None:
            # determine ground truth
            annotation_statistics = cle.statistics_of_background_and_labelled_pixels(annotation, labels)
            classification_gt = annotation_statistics['max_intensity']
            classification_gt[0] = 0
        else:
            classification_gt = None

        feature_list = features.split(' ')

        table, gt = self._select_features(pixel_statistics, feature_list, labels, classification_gt)

        return table, gt

    def _make_touch_matrix(self, labels, touch_matrix = None):
        """Generate an adjacency graph matrix representing touching object.

        Parameters
        ----------
        labels: ndimage
        touch_matrix: ndimage, optional
            will be returned in case not none

        Returns
        -------
        touch_matrix, see [1]

        See Also
        --------
        ..[1] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/neighbors/mesh_between_touching_neighbors.ipynb
        """
        if touch_matrix is None:
            import pyclesperanto_prototype as cle
            touch_matrix = cle.generate_touch_matrix(labels)
        return touch_matrix

    def _make_distance_matrix(self, labels, distance_matrix = None):
        """Generate a matrix with (n+1)*(n+1) elements for a label image with n labels. In this matrix, element (x,y)
        corresponds to the centroid distance between label x and label y.

        Parameters
        ----------
        labels: ndimage(int)
        distance_matrix: ndimage, optional
            will be returned in case not none

        Returns
        -------
        distance_matrix, see [1]

        ..[1] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/neighbors/mesh_with_distances.ipynb
        """
        if distance_matrix is None:
            import pyclesperanto_prototype as cle
            centroids = cle.centroids_of_labels(labels)
            distance_matrix = cle.generate_distance_matrix(centroids, centroids)
            cle.set_column(distance_matrix, 0, 0)
            cle.set_row(distance_matrix, 0, 0)

        return distance_matrix

    def _select_features(self, all_features, features_to_select, labels, ground_truth=None):
        """Provided with all easy-to-determine features, select requested features and calculate the more complicated
        features.

        Parameters
        ----------
        all_features: dict[vector]
        features_to_select: list[str]
        labels: ndimage
        ground_truth: ndimage, optional

        Returns
        -------
        result:Dict[str, np.ndarray]
            list of vectors corresponding to the requested features. The vectors are shaped (n+1) for n labels. The
            first element corresponds to background.
        ground_truth: ndimage
            selected elements of provided ground truth where it's not 0
        """
        import pyclesperanto_prototype as cle
        result = {}
        touch_matrix = None
        distance_matrix = None
        mask = None

        if ground_truth is not None:
            mask = ground_truth > 0

        for key in features_to_select:
            vector = None

            if key in all_features.keys():
                vector = np.asarray([0] + all_features[key])
            elif key == "touching_neighbor_count":
                touch_matrix = self._make_touch_matrix(labels, touch_matrix)
                vector = cle.pull(cle.count_touching_neighbors(touch_matrix))[0]
            elif key == "average_distance_of_touching_neighbors":
                touch_matrix = self._make_touch_matrix(labels, touch_matrix)
                distance_matrix = self._make_distance_matrix(labels, distance_matrix)
                vector = cle.pull(cle.average_distance_of_touching_neighbors(distance_matrix, touch_matrix))[0]
            elif key.startswith("average_distance_of_n_nearest_neighbors="):
                n = int(key.replace("average_distance_of_n_nearest_neighbors=", ""))
                distance_matrix = self._make_distance_matrix(labels, distance_matrix)
                vector = cle.pull(cle.average_distance_of_n_shortest_distances(distance_matrix, n=n))[0]

            if vector is not None:
                if ground_truth is not None:
                    result[key] = np.asarray([vector[mask]])
                else:
                    result[key] = np.asarray([vector])
                # print(key, result[-1])

        if ground_truth is not None:
            return result, ground_truth[mask]
        else:
            return result, None

    def statistics(self):
        return self.classifier.statistics()
