# OPLRAreg

Mixed Integer Piecewise Linear Regression Algorithm with Regularisation

OPLRAreg is a regression technique based on mathematical programming that splits data into separate regions and
identifies independent linear equations for each region.

**This repository only contains the regression algorithm and is meant for general purpose use. For the application on Quantitative Structure-Activity Relationship models, please refer to: https://github.com/KISysBio/qsar-models .**


## Requirements

To run oplrareg, you will need:
  - Python 3 (We have tested versions 3.5 and 3.6)
  - One of the following MIP solver installed in your system:
      - CPLEX 12.6.3 (commercial or academic license)
      - GLPK v4.64 (open source solver)
        - Unix: https://www.gnu.org/software/glpk/
        - Windows: http://winglpk.sourceforge.net/

## Installing

The easiest way to install these packages is using pip:

        pip3 install oplra_reg

Alternatively, you can download [one release](https://github.com/KISysBio/OPLRAreg/releases) and from within the extracted directory, type:

        pip3 install -e .

This will install `oplrareg` package along with its dependencies and it will also create a command `oplrareg`,
which can be used to run OPLRAreg algorithm.

### Dependencies

The following packages will be automatically installed with oplrareg:

    - Pyomo 5.3
    - scikit-learn 0.19.0
    - scipy 0.19.1
    - numpy 1.13.1
    - pandas 0.20.3

## Running OPLRAreg

The command `oplrareg` runs OplraRegularised on the provided data, it accepts tabular and other space delimited files
(.tab, .data, .dat), csv files (.csv) and Excel spreadsheets (.xls and .xlsx).

The only thing to keep in mind is that all columns must be integer/numeric and the outcome variable to be predicted
must be the last column in the data.

Run it with:

        oplrareg --input <filename>
or

        oplrareg -i <filename>

This will execute OPLRAreg with default parameters:

  - lambda = 0.005 (Regularisation parameter. If lambda = 0, no regularisation is used, ideal values are within 0.001 and 0.2)
  - beta = 0.03 (Stopping criteria, lower values will slow down the algorithm)
  - epsilon = 0.01 (Interval between regions, lower values will slow down the algorithm)
  - solver = glpk (If you have a commercial license for CPLEX, use 'cplex' instead)

To test different parameters, pass one of the arguments to the `oplrareg.py` script:

        oplrareg --input <filename> --lambda 0.005 --beta 0.03 --epsilon 0.01 --solver glpk

## Advanced options

It is possible to control which feature will be used to partition the data.
Simply pass the command line argument `--partition_feature` along with the desired column name:

Example:

        oplrareg --input yacht_hydrodynamics.data --partition_feature beam_draught_ratio

By default, OPLRAreg will determine the best number of regions to fit the data but, if desirable, you can control
the number of regions with the parameter `--regions`:

        oplrareg --input yacht_hydrodynamics.data --partition_feature beam_draught_ratio --regions 2
