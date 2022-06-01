import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ._converter import RFC_to_OCL
from ._utils import generate_feature_stack, _read_something_from_opencl_file


class PixelClassifier():
    def __init__(self, opencl_filename = "temp_pixel_classifier.cl", max_depth: int = 2, num_ensembles: int = 100, overwrite_classname:str = None):
        """
        A RandomForestClassifier that converts itself to OpenCL after training.

        Parameters
        ----------
        opencl_filename : str (optional)
        max_depth : int (optional)
        num_ensembles : int (optional)
        overwrite_classname : str
            By setting this parameter, a loaded classifier will be checked if it is of this type.

        See Also
        --------
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        self.FEATURE_SPECIFICATION_KEY = "feature_specification = "
        self.NUM_GROUND_TRUTH_DIMENSIONS_KEY = "num_ground_truth_dimensions = "
        self.CLASSIFIER_CLASS_NAME_KEY = "classifier_class_name = "

        self.max_depth = max_depth
        self.num_ensembles = num_ensembles
        self.output_probability_of_class = -1

        self.opencl_file = opencl_filename
        self.classifier = None

        classname_to_check = self.__class__.__name__
        if overwrite_classname is not None:
            classname_to_check = overwrite_classname

        if classname_to_check != _read_something_from_opencl_file(opencl_filename, self.CLASSIFIER_CLASS_NAME_KEY, classname_to_check):
            raise TypeError("Loading '" + str(classname_to_check) + "' from '" + str(opencl_filename) + "' failed. Wrong classifier type.")

        self.feature_specification = _read_something_from_opencl_file(opencl_filename, self.FEATURE_SPECIFICATION_KEY, "Custom/unkown")
        self.num_ground_truth_dimensions = int(_read_something_from_opencl_file(opencl_filename, self.NUM_GROUND_TRUTH_DIMENSIONS_KEY, 0))

    def train(self, features, ground_truth, image=None, continue_training : bool = False):
        """
        Train a scikit-learn RandomForestClassifier and save it as OpenCL file to disk which
        can be later used for prediction.

        Parameters
        ----------
        features: list of images or str
            Either features are provided as a list of images (or image stacks) or as string.
        ground_truth : ndarray
            2D  or 3D label image with background=0, pixel intensity of label1 = 1, ...
            background pixels will be ignored while training.
        image : ndarray (optional)
            2, 3 or 4D image. If features are provided as string, the feature stack will be generated from this image.
            If the image has more dimensions thant the ground_truth, then the first dimension is assumed to be channels.
            Features will then be generated from the channel images independently.
        continue_training : bool
            if training was done before, it appends the new data and continues training with the whole dataset.
        """
        # make features and convert in the right format
        self.num_ground_truth_dimensions = len(ground_truth.shape)
        features = self._make_features_potentially_multichannel(features, image)

        self.num_features = len(features)
        X, y = self._to_np(features, ground_truth)

        if continue_training:
            if (not hasattr(self, "_X")) or (not hasattr(self, "_y")):
                warnings.warn("Cannot continue training if it wasn't trained before. Will train from scratch instead.")
            else:
                X = np.concatenate((self._X, X),0)
                y = np.concatenate((self._y, y),0)

        self.classifier = RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.num_ensembles, random_state=0)
        self.classifier.fit(X, y)

        self._X = X
        self._y = y

        # save as OpenCL
        self.to_opencl_file(self.opencl_file)

    def predict(self, image=None, features=None):
        """
        Apply an OpenCL-based RandomForestClassifier to a feature stack.

        Parameters
        ----------
        image : ndarray (optional)
            2D or 3D image. If features are provided as string, the feature stack will be generated from this image.
        features: list of images or str
            Either features are provided as a list of images (or image stacks) or as string.

        Returns
        -------
        ndimage, 2D or 3D label image with pixel intensity corresponding to classification
        """
        if features is None:
            features = self.feature_specification

        features = self._make_features_potentially_multichannel(features, image)

        import pyclesperanto_prototype as cle

        if self.output_probability_of_class == -1:
            # produce a label image
            output = cle.create_labels_like(features[0])
        else:
            # produce a probability map
            output = cle.create(features[0].shape) # create float image

        parameters = {}
        for i, f in enumerate(features):
            parameters['in' + str(i)] = f

        parameters['out'] = output

        cle.execute(None, self.opencl_file, "predict", features[0].shape, parameters)

        return output

    def _predict_cpu(self, image=None, features=None):
        """
        Apply a scikit-learn RandomForestClassifier to a feature stack.

        Parameters
        ----------
        features: list of images or str
            Either features are provided as a list of images (or image stacks) or as string.
        image : ndarray (optional)
            2D or 3D image. If features are provided as string, the feature stack will be generated from this image.


        Returns
        -------
        ndimage, 2D or 3D label image with pixel intensity corresponding to classification
        """
        if features is None:
            features = self.feature_specification

        features = self._make_features_potentially_multichannel(features, image)

        if self.classifier is None:
            warnings.warn("Classifier has not been trained.")
            return None

        image = features[0]

        feature_stack, _ = self._to_np(features)

        result_1d = self.classifier.predict(feature_stack)
        result_2d = result_1d.reshape(image.shape)

        return result_2d

    def to_opencl_file(self, filename, extra_information:str = None, overwrite_classname:str = None):
        """
        Save the trained classifier as OpenCL-file.

        Parameters
        ----------
        classifier : scikitlearn.ensemble.RandomForestClassifier
        filename : str

        """
        opencl_code = RFC_to_OCL(self.classifier, self.output_probability_of_class)

        file1 = open(filename, "w")
        file1.write("/*\n")
        file1.write("OpenCL RandomForestClassifier\n")
        if overwrite_classname is not None:
            file1.write(self.CLASSIFIER_CLASS_NAME_KEY + overwrite_classname + "\n")
        else:
            file1.write(self.CLASSIFIER_CLASS_NAME_KEY +  self.__class__.__name__ + "\n")
        file1.write(self.FEATURE_SPECIFICATION_KEY + self.feature_specification + "\n")
        file1.write(self.NUM_GROUND_TRUTH_DIMENSIONS_KEY + str(self.num_ground_truth_dimensions) + "\n")
        file1.write("num_classes = " + str(self.classifier.n_classes_) + "\n")
        file1.write("num_features = " + str(self.num_features) + "\n")
        file1.write("max_depth = " + str(self.max_depth) + "\n")
        file1.write("num_trees = " + str(self.num_ensembles) + "\n")
        if extra_information is not None:
            file1.write(extra_information)
        from apoc import __version__ as version
        file1.write("apoc_version = " + version + "\n")
        file1.write("*/\n")
        file1.write(opencl_code)
        file1.close()

        self.opencl_file = filename

    def _to_np(self, features, ground_truth=None):
        """
        Convert given feature and ground truth images in the right format to be processed by scikit-learn.

        Parameters
        ----------
        features : list of ndarray
        ground_truth : ndarray

        Returns
        -------
        features, ground_truth with each feature and ground_truth a one-dimensional list of numbers
        """
        feature_stack = np.asarray([np.asarray(f).ravel() for f in features]).T
        if ground_truth is None:
            return feature_stack, None
        else:
            # make the annotation 1-dimensional
            ground_truth_np = np.asarray(ground_truth).ravel()

            X = feature_stack
            y = ground_truth_np

            # remove all pixels from the feature and annotations which have not been annotated
            mask = y > 0
            X = X[mask]
            y = y[mask]

            return X, y

    def _make_features_potentially_multichannel(self, features, image):
        """In case multiple images are provided, we interpret these images as list of channels. Hence, features
        will be computed for each image individually. All feature images will be collected in a list. Hence, if
        features A and B are requested for an image I, the final list will be A(I), B(I). If features A and B are
        requested for image I, J and K, the list will be A(I), B(I), A(J), B(J), A(K), B(K).

        Parameters
        ----------
        features: str
        image: ndimage or list[ndimage] or tuple[ndimage]

        Returns
        -------
        list[ndimage]: feature images as list
        """
        if isinstance(image, list) or isinstance(image, tuple) or (hasattr(image, 'shape') and len(image.shape) > self.num_ground_truth_dimensions):
            new_features = []
            for sub_image in image:
                [new_features.append(f) for f in self._make_features(features, sub_image)]
            features = new_features
        else:
            features = self._make_features(features, image)
        return features

    def _make_features(self, features, image = None):
        """
        If features are passed as string, this function will generate a feature stack from the image according to the
        specification in the string.

        Parameters
        ----------
        features : str
        image : ndarray
            2D or 3D image

        Returns
        -------
        list of 2D or 3D images
        """
        if isinstance(features, str):
            self.feature_specification = features
            if image is None:
                raise TypeError("If features are provided as string, an image must be given as well to generate features.")
            features = generate_feature_stack(image, features)

        return features

    def statistics(self):
        """Provide statistics about the trained Random Forest Classifier

        After training or loading a model, this function reads out the decision trees and generates
        statistics from it. It counts for each decision depth how often given features are taken into
        account. It returns two dictionaries. Both dictionaries contain the feature names used during
        training as keys. The values are lists with numbers: The first 'ratios' dictionary contains
        numbers between 0 and 1. The higher that number, the more often is the given feature taken
        into account on the given decision level. The second 'count' dictionary contains the count of
        decisions taking a given feature into account. For example in case the result looks like this:

        A: [0.3, 0.1], [30, 20]
        B: [0.7, 0.9], [70, 180]

        Feature A was taken into account 30% of the decision trees on the first level and in 10% on
        the second level. In case of 100 trees, these are 30 trees on the first level and can be up
        to 20 trees on the second level. Each level doubles the number of available decisions in these
        binary decision trees.

        Returns
        -------
        ratios: dict
        counts: dict
        """
        opencl_filename = self.opencl_file

        # Read information from the header
        feature_specification = _read_something_from_opencl_file(opencl_filename, "feature_specification =", "") \
            .replace(",", " ") \
            .replace("  ", " ") \
            .strip() \
            .split(" ")

        max_depth = int(_read_something_from_opencl_file(opencl_filename, "max_depth = ", "0"))

        # Read the whole forest
        with open(opencl_filename) as f:
            text = f.read()

        # Analyse forest / count existence of "   if(i0" snippets in the code.
        counts = {}
        sums = [0] * max_depth
        for i, feature in enumerate(feature_specification):

            count = []
            for num_spaces in range(max_depth):
                leading_spaces = " " * num_spaces
                num_appearance = text.count("\n" + leading_spaces + "if(i" + str(i))
                sums[num_spaces] += num_appearance
                count.append(num_appearance)
            counts[feature] = np.asarray(count)

        shares = {}
        for i, feature in enumerate(feature_specification):
            shares[feature] = counts[feature] / sums

        return shares, counts
