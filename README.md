# Accelerated Pixel and Object Classifiers (APOC)

[clEsperanto](https://github.com/clEsperanto/pyclesperanto_prototype) meets [sklearn](https://scikit-learn.org/stable/)

## Object segmentation

With a given blobs image and a corresponding annotation...
```python
from skimage.io import imread, imshow
import pyclesperanto_prototype as cle
import numpy as np
import apoc

image = imread('blobs.tif')
imshow(image)
```
![img.png](https://github.com/haesleinhuepf/apoc/raw/main/docs/blobs1.png)
```python
manual_annotations = imread('annotations.tif')
imshow(manual_annotations, vmin=0, vmax=3)
```
![img.png](https://github.com/haesleinhuepf/apoc/raw/main/docs/blobs_annotations1.png)

... objects can be segmented ([see full example](https://github.com/haesleinhuepf/apoc/blob/main/demo/demo_object_segmenter.ipynb)):
```python
# define features: original image, a blurred version and an edge image
features = features = apoc.PredefinedFeatureSet.medium_quick.value

clf = apoc.ObjectSegmenter(opencl_filename='object_segmenter.cl', positive_class_identifier=2)
clf.train(features, manual_annotations, image)

segmentation_result = clf.predict(image=image)
cle.imshow(segmentation_result, labels=True)
```
![img.png](https://github.com/haesleinhuepf/apoc/raw/main/docs/blobs_segmentation1.png)

## Object classification

With a given annotation, blobs can also be classified according to their shape ([see full example](https://github.com/haesleinhuepf/apoc/blob/main/demo/demo_object_segmenter.ipynb)).
```python
features = 'area,mean_max_distance_to_centroid_ratio,standard_deviation_intensity'

# Create an object classifier
classifier = apoc.ObjectClassifier("object_classifier.cl")

# train it
classifier.train(features, segmentation_result, annotation, image)

# determine object classification
classification_result = classifier.predict(segmentation_result, image)

imshow(classification_result)
```
![img.png](https://github.com/haesleinhuepf/apoc/raw/main/docs/object_classification_result1.png)

## More detailed examples

* [Object segmentation](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_object_segmenter.ipynb)  
* [Object classification](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_object_classification.ipynb)  
* [Pixel classifier (including benchmarking)](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/benchmarking_pixel_classifier.ipynb).
* [Continue training of pixel classifiers using multiple training image pairs](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demp_pixel_classifier_continue_training.ipynb)
* [Generating custom feature stacks](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/feature_stacks.ipynb)


## Installation

You can install `apoc` via [pip]. Note: you also need [pyopencl](https://documen.tician.de/pyopencl/).

    conda install pyopencl
    pip install apoc

## Contributing

Contributions are very welcome. Tests can be run with `pytest`, please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the BSD-3 license,
"apoc" is free and open source software

## Issues

If you encounter any problems, please [open a thread on image.sc](https://image.sc) along with a detailed description and tag [@haesleinhuepf](https://github.com/haesleinhuepf).
