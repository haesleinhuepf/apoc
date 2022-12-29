from ._object_classifier import ObjectClassifier
from ._utils import _read_something_from_opencl_file

class ObjectSelector():
    def __init__(self, opencl_filename="temp_object_selector.cl", max_depth: int = 2, num_ensembles: int = 100, positive_class_identifier : int = 2):
        """
        A RandomForestClassifier for object selection according to a label classification
        that converts itself to OpenCL after training. The selector uses an ObjectClassifer under the hood
        The result is a new label image containing only objects of the specified positive class.

        Parameters
        ----------
        opencl_filename : str (optional)
        max_depth : int (optional)
        num_ensembles : int (optional)
        positive_class_identifier:int (optional)

        See Also
        --------
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        self.POSITIVE_CLASS_IDENTIFIER_KEY = "positive_class_identifier = "

        self.positive_class_identifier_from_file = int(
            _read_something_from_opencl_file(opencl_filename, self.POSITIVE_CLASS_IDENTIFIER_KEY,
                                             positive_class_identifier))
        self.positive_class_identifier = positive_class_identifier
        self.opencl_filename = opencl_filename
        self.classifier = ObjectClassifier(
            opencl_filename=opencl_filename,
            max_depth=max_depth,
            num_ensembles=num_ensembles,
            overwrite_classname=self.__class__.__name__)

    def train(self, features: str, labels, sparse_annotation, image=None, continue_training: bool = False):
        """
        Train a classifier that can select objects according to their intensity, size and shape.

        See also
        --------
        .. ObjectClassifier.train()
        """

        self.positive_class_identifier_from_file = self.positive_class_identifier
        self.classifier.train(features, labels, sparse_annotation, image, continue_training=continue_training)
        self.to_opencl_file(self.opencl_filename, overwrite_classname=self.__class__.__name__)

    def predict(self, labels, image=None):
        """
        Apply the selector to a given image.

        See also
        --------
        .. ObjectClassifier.predict()
        """
        import pyclesperanto_prototype as cle
        self.positive_class_identifier = self.positive_class_identifier_from_file

        classification_image = self.classifier.predict(labels, image)
        values_vector = cle.read_intensities_from_map(labels, classification_image)
        return cle.exclude_labels_with_values_not_equal_to_constant(values_vector,
                                                                    labels,
                                                                    constant=self.positive_class_identifier)

    def to_opencl_file(self, filename, extra_information: str = None, overwrite_classname: str = None):
        """Save the classifier to an OpenCL-file. The file will also contain the selected class identifier.

        See Also
        --------
        .. PixelClassifier.to_opencl_file()
        """
        extra = self.POSITIVE_CLASS_IDENTIFIER_KEY + str(self.positive_class_identifier) + "\n"
        if extra_information is not None:
            extra = extra + extra_information
        if overwrite_classname is None:
            overwrite_classname = self.__class__.__name__
        return self.classifier.to_opencl_file(filename=filename, extra_information=extra, overwrite_classname=overwrite_classname)

    def feature_importances(self):
        """Provide feature importances about the trained Random Forest Classifier

        The values are provided as dictionary {feature_name:portion_importance}.

        See also
        --------
        ..[0] https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        """
        return self.classifier.feature_importances()

    def statistics(self):
        return self.classifier.statistics()
