from ._pixel_classifier import PixelClassifier

class ProbabilityMapper(PixelClassifier):
    def __init__(self, opencl_filename = "temp_probability_mapper.cl", max_depth: int = 2, num_ensembles: int = 10, output_probability_of_class: int = 0):
        super().__init__(opencl_filename=opencl_filename, max_depth=max_depth, num_ensembles=num_ensembles)
        self.output_probability_of_class = output_probability_of_class
