# Accelerated Pixel and Object Classifiers (APOC)
[![License](https://img.shields.io/pypi/l/apoc.svg?color=green)](https://github.com/haesleinhuepf/apoc/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/apoc.svg?color=green)](https://pypi.org/project/apoc)
[![Python Version](https://img.shields.io/pypi/pyversions/apoc.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/apoc/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/apoc/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/apoc/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/apoc)
[![Development Status](https://img.shields.io/pypi/status/apoc.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![DOI](https://zenodo.org/badge/412505712.svg)](https://zenodo.org/badge/latestdoi/412505712)

[clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype) meets [scikit-learn](https://scikit-learn.org/stable/) to classify pixels and objects in images, on a [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) using [OpenCL](https://www.khronos.org/opencl/).
This repository contains the backend for python developers. User-friendly plugins for [Fiji](https://fiji.sc) and [napari](https://napari.org) can be found here:
* [napari-accelerated-pixel-and-object-classification](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification)
* [clijx-accelerated-pixel-and-object-classification](https://github.com/clij/clijx-accelerated-pixel-and-object-classification)

For training classifiers from pairs of image and label-mask folders, please see 
[this notebook](https://github.com/haesleinhuepf/apoc/blob/main/demo/train_on_folders.ipynb).

![](https://github.com/clij/clijx-accelerated-pixel-and-object-classification/raw/main/docs/img.png)

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
features = apoc.PredefinedFeatureSet.medium_quick.value

# Training
clf = apoc.ObjectSegmenter(opencl_filename='object_segmenter.cl', positive_class_identifier=2)
clf.train(features, manual_annotations, image)

# Prediction
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

# Training
classifier.train(features, segmentation_result, annotation, image)

# Prediction / determine object classification
classification_result = classifier.predict(segmentation_result, image)

imshow(classification_result)
```
![img.png](https://github.com/haesleinhuepf/apoc/raw/main/docs/object_classification_result1.png)

## More detailed examples

* [Object segmentation](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_object_segmenter.ipynb)  
* [Object classification](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_object_classification.ipynb)  
* [Object classification based on custom measurement tables](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/table_row_classification.ipynb)  
* [Pixel classifier (single channel, including benchmarking)](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_pixel_classifier_single_channel.ipynb).
* [Pixel classifier (multi channel, including benchmarking)](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_pixel_classifier_multi_channel.ipynb).
* [Output probability maps](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demo_probability_mapper.ipynb)  
* [Continue training of pixel classifiers using multiple training image pairs](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/demp_pixel_classifier_continue_training.ipynb)
* [Generating custom feature stacks](https://nbviewer.jupyter.org/github/haesleinhuepf/apoc/blob/main/demo/feature_stacks.ipynb)


## Installation

You can install `apoc` using pip. Note: you need to install [pyopencl](https://documen.tician.de/pyopencl/) in advance using conda:

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
