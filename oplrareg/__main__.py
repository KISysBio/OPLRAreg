#!/usr/bin/env python
"""Collect command-line options in a dictionary"""

import pandas as pd
import logging
import os

from oplrareg import *
from sklearn import preprocessing

logger = logging.getLogger(__name__)


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == "-" or argv[0][0] == "--":  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


def parse_input(input_file):
    if input_file.endswith(".tab") or input_file.endswith(".data"):
        logger.info("Reading tabular data")
        contents = pd.read_csv(input_file, skiprows=1, sep=r"\s+")
    elif input_file.endswith(".csv"):
        logger.info("Reading .csv file")
        contents = pd.read_csv(input_file)
    elif input_file.endswith(".xls") or input_file.endswith(".xlsx"):
        logger.info("Reading spreadsheet")
        contents = pd.read_excel(input_file)
        raise ValueError("Input file extension is not recognised:  %s" % input_file)
    X = contents.iloc[:, :-1]
    y = contents.iloc[:, -1]

    return X, y


def main():
    from sys import argv

    myargs = getopts(argv)
    if "--input" in myargs or "-i" in myargs:
        if "--input" in myargs:
            input_file = myargs["--input"]
        else:
            input_file = myargs["-i"]
        X, y = parse_input(input_file)
    else:
        msg = (
            "Please inform an input file with --input or -i argument.\n"
            + "\tExample: \n\t\toplrareg --input yacht_hydrodynamics.data\n"
            + "\t\toplrareg -i yacht_hydrodynamics.data"
        )
        raise ValueError(msg)

    min_max_scaler = preprocessing.MinMaxScaler()
    import ipdb; ipdb.set_trace()
    scaled_X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

    params = {}
    fit_params = {}

    if "-l" in myargs:  # Example usage.
        params["lam"] = float(myargs["-l"])
    elif "--lambda" in myargs:
        params["lam"] = float(myargs["--lambda"])

    if "-e" in myargs:  # Example usage.
        params["epsilon"] = float(myargs["-e"])
    elif "--epsilon" in myargs:
        params["epsilon"] = float(myargs["--epsilon"])

    if "-b" in myargs:  # Example usage.
        params["beta"] = float(myargs["-b"])
    elif "--beta" in myargs:
        params["beta"] = float(myargs["--beta"])

    if "-s" in myargs:  # Example usage.
        params["solver_name"] = myargs["-b"]
    elif "--solver" in myargs:
        params["solver_name"] = myargs["--solver"]

    if "-r" in myargs:  # Example usage.
        params["exact_number_regions"] = int(myargs["-r"])
    elif "--regions" in myargs:
        params["exact_number_regions"] = int(myargs["--regions"])

    if "-pf" in myargs:  # Example usage.
        fit_params["f_star"] = myargs["-f"]
    elif "--partition-feature" in myargs:
        fit_params["f_star"] = myargs["--partition-feature"]

    oplra_reg = OplraRegularised()
    if len(params) > 0:
        oplra_reg.set_params(**params)

    if len(fit_params) > 0:
        oplra_reg.fit(scaled_X, y, f_star=fit_params["f_star"])
    else:
        oplra_reg.fit(scaled_X, y)

    coeffs, breakpoints = oplra_reg.get_model_info()

    print("BREAKPOINTS:")
    print(breakpoints)
    print()

    print("COEFFICIENTS:")
    print(coeffs)
    print()

    print("MEAN ABSOLUTE ERROR: %f" % oplra_reg.final_model.mae.value)

    input_name = os.path.splitext(input_file)[0]

    coeffs.to_csv(input_name + "_coefficients.csv", index=False)
    breakpoints.to_csv(input_name + "_breakpoints.csv", index=False)


if __name__ == "__main__":
    main()
