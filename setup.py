import setuptools

setuptools.setup(
    name="pyrankability",
    version="0.0.11",
    author="Paul Anderson",
    author_email="pauleanderson@gmail.com",
    description="Rankability Toolbox",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/pyrankability",
    packages=setuptools.find_packages(),
    install_requires=[ "networkx", "numpy", "pandas" , "scipy", "dask", "toolz", "joblib", "distributed", "sklearn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
