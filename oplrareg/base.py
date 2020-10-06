from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from .models import OplraPyomoModel
from .solvers import get_solver_definition


class BaseOplraEstimator(BaseEstimator, RegressorMixin):
    """Superclass of estimators used in this package
    """

    def __init__(
        self, algorithm_name, algorithm_version, lam, epsilon, beta, solver_name
    ):
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self.lam = lam
        self.epsilon = epsilon
        self.beta = beta
        self.solver_name = solver_name
        self.solver_def = get_solver_definition(solver_name)

    def get_model_info(self):
        raise NotImplementedError("Function should be implemented in algorithm's class")

    def run_model(self, X, y, number_regions, f_star=None, selected_features=None):
        pyomo_model = build_piecewise_model(
            X, y, self.lam, self.epsilon, number_regions, f_star, selected_features
        )
        termination_condition = self.solver_def.solve(pyomo_model, verbose=False)
        return pyomo_model, termination_condition


def build_piecewise_model(
    data,
    target,
    lam,
    epsilon,
    number_regions,
    f_star=None,
    selected_features=None,
    n_max=None,
):
    """
        Helper function to create a OplraPyomoModel
        :param data:
        :param target:
        :param number_regions:
        :param f_star:
        :param selected_features:
        :param n_max:
        :return:
    """

    return OplraPyomoModel(
        data,
        target,
        number_regions,
        lam=lam,
        epsilon=epsilon,
        f_star=f_star,
        selected_features=selected_features,
    )
