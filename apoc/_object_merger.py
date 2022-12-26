from typing import Dict, List, Union, Optional
import pyclesperanto_prototype as cle

from ._probability_mapper import PixelClassifier


class ObjectMerger:
    def __init__(
            self,
            opencl_filename="temp_label_merger.cl",
            max_depth: int = 2, num_ensembles: int = 100,
            overwrite_classname: Optional[str] = None,
    ):
        """A RandomForestClassifier for merging touching labels in a
        label image according to an annotation and features extracted
        from the borders of touching labels.

        annotation = 1 : merge
        annotation = 2 : do not merge

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
        self.classifier = PixelClassifier(
            opencl_filename=opencl_filename,
            max_depth=max_depth,
            num_ensembles=num_ensembles,
            overwrite_classname=self.__class__.__name__)

    def train(self, features: str, labels, sparse_annotation, image=None):
        """Train the classifier to be able to merge labels as annotated.

        annotation = 1 : merge
        annotation = 2 : do not merge
        annotation = 0 : unknown / not decided

        Note: for technical reasons, the intensity image is internal converted to integer.
        Thus, you should make sure its intensities are in a reasonable range (not 0-1).

        Parameters
        ----------
        features: Space separated string containing those:
            'mean_touch_intensity',
            'touch_portion',
            'touch_count'
        labels: label image
        sparse_annotation: label image with annotations
        image: intensity image (optional)
            must be provided if feature `mean_touch_intensity` is specified
        """
        self.classifier.feature_specification = features.replace(",", " ")
        # remove too many spaces
        while "  " in self.classifier.feature_specification:
            self.classifier.feature_specification = self.classifier.feature_specification.replace("  ", " ")
        self.classifier.feature_specification = self.classifier.feature_specification.strip()

        # extract features
        feature_images = self._make_features(self.classifier.feature_specification, labels, image)

        # determine ground truth
        should_touch_matrix = cle.generate_should_touch_matrix(labels, sparse_annotation == 1)
        should_not_touch_matrix = cle.generate_should_touch_matrix(labels, sparse_annotation == 2)
        ground_truth_matrix = should_touch_matrix + should_not_touch_matrix * 2

        feature_specification_backup = self.classifier.feature_specification
        self.classifier.train("original", ground_truth_matrix, image=feature_images)
        self.classifier.feature_specification = feature_specification_backup
        self.classifier.to_opencl_file(self.classifier.opencl_file, overwrite_classname=self.classifier.classname)

    def predict(self, labels, image=None):
        """
        Apply classifier to label image and return a label image with merged labels.

        Note: If an intensity image was provided for training, it must be provided here as well.

        Parameters
        ----------
        labels: label image
        image: intensity image (optional)

        Returns
        -------
        Label image with merged labels
        """
        # extract features
        feature_images = self._make_features(self.classifier.feature_specification, labels, image)

        feature_specification_backup = self.classifier.feature_specification
        predicted_matrix = self.classifier.predict(image=feature_images, features="original")
        predicted_matrix = predicted_matrix == 1
        self.classifier.feature_specification = feature_specification_backup

        # we will only merge those which are touching
        touch_matrix = cle.generate_touch_matrix(labels)
        merge_matrix = cle.binary_and(touch_matrix, predicted_matrix)

        # ignore background
        cle.set_column(merge_matrix, 0, 0)
        cle.set_row(merge_matrix, 0, 0)

        return cle.merge_labels_according_to_touch_matrix(labels, merge_matrix)

    def _make_features(self, feature_specification, labels, image):
        """
        Produce feature images from label image and intensity image according to specification.
        See train() for the available feature list.
        """
        features = []
        for f in feature_specification.split(" "):
            if f == "mean_touch_intensity":
                features.append(cle.generate_touch_mean_intensity_matrix(cle.asarray(image).astype(int), labels))
            elif f == "touch_portion":
                features.append(cle.generate_touch_portion_matrix(labels))
            elif f == "touch_count":
                features.append(cle.generate_touch_count_matrix(labels))
            else:
                raise ValueError("Unknown feature: " + f)

        return features