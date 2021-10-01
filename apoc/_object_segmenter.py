from ._pixel_classifier import PixelClassifier

class ObjectSegmenter(PixelClassifier):

    
    def __init__(self, opencl_filename = "temp.cl", max_depth: int = 2, num_ensembles: int = 10, positive_class_identifi : int = 2):
        super(opencl_filename=opencl_filename, max_depth =max_depth, num_ensembles=num_ensembles)
        self.positive_class_identifier = positive_class_identifier


    def to_opencl_file(self, filename, extra_information:str = None):
        super().to_opencl_file(filename, "positive_class_identifier = " + self.positive_class_identifier + "\n" +  extra_information)

    def predict(self, features=None, image=None):
        result = super().predict(features=features, image=image)

        import pyclesperanto_prototype as cle
        binary = cle.equal_constant(result, constant=self.positive_class_identifier)
        
        return cle.connected_components_labeling_diamond(binary)


