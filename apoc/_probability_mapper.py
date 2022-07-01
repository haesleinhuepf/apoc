from ._pixel_classifier import PixelClassifier

class ProbabilityMapper(PixelClassifier):
    def __init__(self, opencl_filename = "temp_probability_mapper.cl", max_depth: int = 2, num_ensembles: int = 100, output_probability_of_class: int = 0):
        """The ProbabilityMapper is a pixel classifier that does return the probability image for one specific class
        instead of returning a class-image. The training is the same as in PixelClassifier, just the result is not a
        label image of type int, but an probability image of type float.

        Parameters
        ----------
        opencl_filename: str
        max_depth: int
        num_ensembles: int
        output_probability_of_class: int
            class identifier of which the probability should be exported.

        See Also
        --------
        .. PixelClassifier.__init__()
        """
        super().__init__(opencl_filename=opencl_filename, max_depth=max_depth, num_ensembles=num_ensembles)
        self.output_probability_of_class = output_probability_of_class
