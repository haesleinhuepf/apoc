from ._converter import RFC_to_OCL
from ._pixel_classifier import PixelClassifier
from ._object_segmenter import ObjectSegmenter
from ._object_classifier import ObjectClassifier
from ._probability_mapper import ProbabilityMapper
from ._utils import generate_feature_stack
from ._utils import erase_classifier
from ._utils import train_classifier_from_image_folders
from ._feature_sets import PredefinedFeatureSet

__version__ = "0.6.4"
