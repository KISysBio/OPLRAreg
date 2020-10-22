import pandas as pd

from collections import Counter
from pyomo.opt import TerminationCondition

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot

from .models import *
from .base import *


class OplraRegularised(BaseOplraEstimator):
    """Runs Oplra with regularisation

    Attributes:
        lam          (float) : Regularisation parameter
        epsilon      (float) : MIP parameter, small number to separate the breakpoints
        beta         (float) : Stopping criteria, usually set to 0.03. Algorithm will stop when the improvement
                                in the objective function is no more than beta.
        exact_number_regions (int) : If specified, runs Oplra to the the number of regions passed to this argument
        solver_name     (str) : Name of a supported solver (CPLEX or GLPK)
    """

    def __init__(
        self,
        lam=0.005,
        epsilon=0.01,
        beta=0.03,
        exact_number_regions=None,
        solver_name="glpk",
    ):
        super().__init__("OplraRegularised", "v1", lam, epsilon, beta, solver_name)
        self.final_model = None
        if exact_number_regions is None:
            self.exact_number_regions = None
            self.beta = beta
        else:
            self.beta = None
            self.exact_number_regions = exact_number_regions

    def predict(self, X):
        """Predicts outcome of new data

        :param X: data to predict outcome
        :return: list with predictions for each sample in X
        """

        # It is necessary to run `fit` function before `predict`
        check_is_fitted(self, "final_model")

        samples = range(X.shape[0])

        # Defines the regions of the samples
        if self.final_model.number_regions > 1:
            if type(X) == pd.DataFrame:
                ts_region = self.final_model.get_regions(X.loc[:, self.final_model.fStar])
            else:
                # Then it is a numpy array

                fStar_idx = None
                for idx, f in enumerate(self.final_model.f):
                    if f == self.final_model.fStar:
                        fStar_idx = idx
                        break

                ts_region = self.final_model.get_regions(X[:, fStar_idx])

        else:
            ts_region = np.zeros(len(samples))

        counter_region = Counter(ts_region)
        final_predictions = np.zeros(len(samples))
        coefficients = self.final_model.get_coefficients()
        intercepts = self.final_model.get_intercepts()

        if len(counter_region) == 1:

            which_region = counter_region.most_common(1)[0][0]

            final_predictions = (
                safe_sparse_dot(X, coefficients[which_region], dense_output=True) + intercepts[which_region]
            )
        else:
            for region, counter_region in counter_region.items():
                samples_in_region = np.argwhere(ts_region == region).transpose()[0]
                if type(X) == pd.DataFrame:
                    X_samples = X.iloc[samples_in_region]
                else:
                    X_samples = X[samples_in_region, ]

                final_predictions[samples_in_region] = (
                    safe_sparse_dot(
                        X_samples,
                        coefficients[region],
                        dense_output=True,
                    )
                    + intercepts[region]
                )

        return final_predictions

    def fit(self, X, y, f_star=None, verbose=True):
        """

        :param X:
        :param y:
        :param f_star
        :param verbose
        :return:
        """

        if verbose:
            print("%s\n########## R = 1" % self)

        simple_linear_model, termination_condition = self.run_model(X, y, 1)
        selected_features = simple_linear_model.get_selected_features()

        print("SELECTED FEATURES:")
        print(selected_features)
        print()

        if len(selected_features) == 0:
            self.final_model = simple_linear_model
            return self

        # Solve for exact number of regions == 1 or when exact number of regions >= 2 when f* was defined beforehand
        if self.beta is None or self.exact_number_regions == 1:
            self.final_model = simple_linear_model
            return self
        elif f_star is not None:
            number_regions = self.exact_number_regions
            tmp_model, termination_condition = self.run_model(
                X, y, number_regions, f_star, selected_features
            )
            self.final_model = tmp_model
            return self

        # ---- PIECEWISE MODEL R = 2 ---- #
        number_regions = 2
        best_model = None
        count = 1
        print(f_star)

        if f_star is None:
            # ---- Find best f* for 2 regions ---- #
            for f_star in selected_features:
                if verbose:
                    print(
                        "\n%s\n########## R = %d f* = %s (loop %d/%d) ###########"
                        % (self, number_regions, f_star, count, len(selected_features))
                    )

                tmp_model, termination_condition = self.run_model(
                    X, y, number_regions, f_star, selected_features
                )
                count += 1
                if termination_condition != TerminationCondition.infeasible and (
                    best_model is None or tmp_model.z.value < best_model.z.value
                ):
                    best_model = tmp_model
        else:
            if verbose:
                print(
                    "\n%s\n########## R = %d f* = %s ###########"
                    % (self, number_regions, f_star)
                )

            tmp_model, termination_condition = self.run_model(
                X, y, number_regions, f_star, selected_features
            )
            if termination_condition != TerminationCondition.infeasible and (
                best_model is None or tmp_model.z.value < best_model.z.value
            ):
                best_model = tmp_model
            else:
                raise ValueError("Unable to find a partition for f* = %s" % f_star)

        # Solve for exact number of regions >= 2 when f* was not defined beforehand
        if self.beta is None or self.exact_number_regions == 2:
            self.final_model = best_model
        elif self.exact_number_regions is not None:
            number_regions = self.exact_number_regions
            tmp_model, termination_condition = self.run_model(
                X, y, number_regions, f_star, selected_features
            )
            self.final_model = tmp_model
        # Otherwise, keep increasing the number of regions until best model is achieved
        else:
            f_star = best_model.fStar
            previousZ = float("inf")
            currentZ = best_model.z.value

            while currentZ < previousZ * (1 - self.beta):
                number_regions += 1
                if verbose:
                    print(
                        "\n%s\n########## R = %d f* = %s ###########"
                        % (self, number_regions, f_star)
                    )

                tmp_model, termination_condition = self.run_model(
                    X, y, number_regions, f_star, selected_features
                )
                if termination_condition == TerminationCondition.infeasible:
                    break
                previousZ, currentZ = currentZ, tmp_model.z.value
                if currentZ < previousZ * (1 - self.beta):
                    best_model = tmp_model

            self.final_model = best_model
        return self

    def get_model_info(self):
        coefficients = pd.DataFrame()

        featureNames = self.final_model.get_selected_features()
        cols = np.array("region")
        cols = np.append(cols, featureNames)
        cols = np.append(cols, "B")

        for region in range(self.final_model.number_regions):
            values = {"region": region}
            for f in featureNames:
                values.update({f: self.final_model.W[region, f].value})
            values.update({"B": self.final_model.B[region].value})
            coefficients = coefficients.append(
                pd.DataFrame(data=values, index=np.arange(1), columns=cols),
                ignore_index=True,
            )

        if self.final_model.number_regions == 1:
            breakpointsDF = pd.DataFrame(
                data={"fStar": None, "breakpoints": None}, index=np.arange(1)
            )
        else:
            breakpointsDF = pd.DataFrame(
                {
                    "fStar": self.final_model.fStar,
                    "breakpoints": self.final_model.get_breakpoints(),
                }
            )

        return coefficients, breakpointsDF
