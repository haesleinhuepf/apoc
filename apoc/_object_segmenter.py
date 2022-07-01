from ._pixel_classifier import PixelClassifier
from ._utils import _read_something_from_opencl_file

class ObjectSegmenter(PixelClassifier):

    
    def __init__(self, opencl_filename = "temp_object_segmenter.cl", max_depth: int = 2, num_ensembles: int = 100, positive_class_identifier : int = 2):
        """
        A Random Forest Classifier that classifies pixels and afterwards selects a single class and applies connected
        component labeling to all pixels of that class.

        Parameters
        ----------
        opencl_filename: str, file-path
            The classifier will be loaded from and/or stored to this file.
        max_depth: int
            maximum tree depth of the trees in the random forest classifier
        num_ensembles: int
            number of trees
        positive_class_identifier: int
            The class that identifies objects which should be returned as label image.
        """
        super().__init__(opencl_filename=opencl_filename, max_depth =max_depth, num_ensembles=num_ensembles)

        self.POSITIVE_CLASS_IDENTIFIER_KEY = "positive_class_identifier = "

        self.positive_class_identifier_from_file = int(_read_something_from_opencl_file(opencl_filename, self.POSITIVE_CLASS_IDENTIFIER_KEY, positive_class_identifier))
        self.positive_class_identifier = positive_class_identifier

    def train(self, features, ground_truth, image=None, continue_training : bool = False):
        """Train the classifier with.

        See Also
        --------
        .. PixelClassifier.train()
        """
        self.positive_class_identifier_from_file = self.positive_class_identifier
        super().train(features, ground_truth, image, continue_training=continue_training)

    def to_opencl_file(self, filename, extra_information:str = None):
        """Save the classifier to an OpenCL-file. The file will also contain the selected class identifier.

        See Also
        --------
        .. PixelClassifier.to_opencl_file()
        """
        extra = self.POSITIVE_CLASS_IDENTIFIER_KEY + str(self.positive_class_identifier) + "\n"
        if extra_information is not None:
            extra = extra + extra_information

        return super().to_opencl_file(filename=filename, extra_information=extra)

    def predict(self, image=None, features=None):
        """Apply the classifier + class selection + connected component labeling to a given image.

        See Also
        --------
        .. PixelClassifier.predict()
        """
        self.positive_class_identifier = self.positive_class_identifier_from_file
        result = super().predict(features=features, image=image)

        import pyclesperanto_prototype as cle
        binary = cle.equal_constant(result, constant=self.positive_class_identifier)
        
        return cle.connected_components_labeling_diamond(binary)


