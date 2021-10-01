import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oclrfc",
    version="0.4.1",
    author="haesleinhuepf",
    author_email="robert.haase@tu-dresden.de",
    description="OpenCL-based Random Forest Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haesleinhuepf/oclrfc",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["numpy!=1.19.4", "scikit-learn>=0.24.2", "pyclesperanto-prototype>=0.8.0"],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],
)
