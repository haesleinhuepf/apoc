import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="apoc",
    version="0.12.0",
    author="haesleinhuepf",
    author_email="robert.haase@tu-dresden.de",
    description="Accelerated Pixel and Object Classifiers based on OpenCL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haesleinhuepf/apoc",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy>=2.0", "scikit-learn>=1.5.0", "pyclesperanto-prototype>=0.24.1", "pandas>=2.0.0"],
    python_requires='>=3.11',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],
)
