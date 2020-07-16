import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ebano-express",
    version="0.0.1",
    author="Francesco Ventura",
    author_email="francescoventura.183464@gmail.com",
    description="Explain black-box DCNN predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EBAnO-Ecosystem/EBAnO-Express.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)