from ._pixel_classifier import PixelClassifier
from ._utils import _read_something_from_opencl_file

class ObjectSegmenter(PixelClassifier):

    
    def __init__(self, opencl_filename = "temp_object_segmenter.cl", max_depth: int = 2, num_ensembles: int = 10, positive_class_identifier : int = 2):
        super().__init__(opencl_filename=opencl_filename, max_depth =max_depth, num_ensembles=num_ensembles)

        self.POSITIVE_CLASS_IDENTIFIER_KEY = "positive_class_identifier = "

        self.positive_class_identifier = int(_read_something_from_opencl_file(opencl_filename, self.POSITIVE_CLASS_IDENTIFIER_KEY, positive_class_identifier))

    def to_opencl_file(self, filename, extra_information:str = None):

        extra = self.POSITIVE_CLASS_IDENTIFIER_KEY + str(self.positive_class_identifier) + "\n"
        if extra_information is not None:
            extra = extra + extra_information

        return super().to_opencl_file(filename=filename, extra_information=extra)

    def predict(self, features=None, image=None):
        result = super().predict(features=features, image=image)

        import pyclesperanto_prototype as cle
        binary = cle.equal_constant(result, constant=self.positive_class_identifier)
        
        return cle.connected_components_labeling_diamond(binary)


