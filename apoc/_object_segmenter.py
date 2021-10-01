from ._pixel_classifier import PixelClassifier
import os

class ObjectSegmenter(PixelClassifier):

    
    def __init__(self, opencl_filename = "temp.cl", max_depth: int = 2, num_ensembles: int = 10, positive_class_identifier : int = 2):
        super().__init__(opencl_filename=opencl_filename, max_depth =max_depth, num_ensembles=num_ensembles)

        self.POSITIVE_CLASS_IDENTIFIER_KEY = "positive_class_identifier = "

        self.positive_class_identifier = self._get_positive_class_identifier_from_opencl_file(opencl_filename)
        if self.positive_class_identifier is None:
            self.positive_class_identifier = positive_class_identifier

    def _get_positive_class_identifier_from_opencl_file(self, opencl_filename):
        """
        Reads the positive class identifier from an OpenCL file. It's typically saved there in the header after training.

        Parameters
        ----------
        opencl_filename : str

        Returns
        -------
        str, see _utils.generate_feature_stack
        """
        if not os.path.exists(opencl_filename):
            return

        with open(opencl_filename) as f:
            line = ""
            count = 0
            while line != "*/" and line is not None and count < 25:
                count = count + 1
                line = f.readline()
                if line.startswith(self.POSITIVE_CLASS_IDENTIFIER_KEY):
                    return int(line.replace(self.POSITIVE_CLASS_IDENTIFIER_KEY, "").replace("\n",""))

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


