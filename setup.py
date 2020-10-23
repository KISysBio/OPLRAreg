from setuptools import setup, find_packages

setup(
    name="oplra_reg",
    description="Mixed Integer Piecewise Linear Regression with Regularisation",
    long_description=(
        "OPLRAreg is a regression technique based on mathematical programming that splits "
        "data into separate regions and identifies independent linear equations for each region."
    ),
    version="0.3",
    py_modules=["oplrareg"],
    author="Jonathan Cardoso Silva",
    author_email="jonathan.car.silva@gmail.com",
    url="https://github.com/KISysBio/OPLRAreg",
    url_download="https://github.com/KISysBio/OPLRAreg/archive/v0.2.1.tar.gz",
    keywords=[""],
    install_requires=[
        "numpy >= 1.13",
        "scipy >= 0.19.1",
        "pandas >= 0.20.3",
        "scikit-learn >= 0.19.0",
        "pyomo >= 5.3",
    ],
    entry_points={"console_scripts": ["oplrareg = oplrareg.__main__:main"]},
    packages=find_packages(),
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
