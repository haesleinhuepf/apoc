# oclrfc

[cle](https://github.com/clEsperanto/pyclesperanto_prototype) meets [sklearn](https://scikit-learn.org/stable/)

To see OpenCL-based Random Forest Classifiers in action, check out the 
[demo-notebook](https://nbviewer.jupyter.org/github/haesleinhuepf/oclrfc/blob/master/demo/demo.ipynb).
For optimal performance and classification quality, it is recommended to 
[generate feature stacks](https://nbviewer.jupyter.org/github/haesleinhuepf/oclrfc/blob/master/demo/feature_stacks.ipynb)
that fit well to the the image data you would like to process.

## Installation

You can install `oclrfc` via [pip]. Note: you also need [pyopencl](https://documen.tician.de/pyopencl/).

    conda install pyopencl
    pip install oclrfc

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the BSD-3 license,
"oclrfc" is free and open source software

## Issues

If you encounter any problems, please [open a thread on image.sc](https://image.sc) along with a detailed description and tag [@haesleinhuepf](https://github.com/haesleinhuepf).
