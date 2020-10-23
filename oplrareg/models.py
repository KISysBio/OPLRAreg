import numpy as np
import pandas as pd

from pyomo.core.base import (
    ConcreteModel,
    Var,
    Set,
    NonNegativeReals,
    Binary,
    Constraint,
    Objective,
    minimize,
    Reals,
)


class OplraPyomoModel(ConcreteModel):
    """A Pyomo concrete model that defines OPLRA mathematical programming model

    Attributes:
        data         (DataFrame): data frame containing samples and features
        target         (object) : a list or pandas Series of the target variables of regression
        number_regions         (int) : if specified, runs Oplra to the exact number of regions
        lam             (float) : lambda regularisation parameter
        epsilon         (float) : MIP epsilon parameter (usually set at 0.01)
        nMax              (int) : if specified, add constraints that limit the maximum number of features to be selected
        fStar             (str) : partition feature f* to be used in the model. Only valid for noRegions >= 2
        selectedFeatures (list) : subset of features to be used in the MIP model. Usually defined after running OPLRA
                                    for noRegions == 1 and selecting those features with nonzero coefficients
        sampleWeight     (list) : Weights assigned to the samples
        isSimpleLinear   (bool) : if True, sets all features to selected. (Probably will be deprecated)
    """

    def __init__(
        self,
        data,
        target,
        number_regions,
        lam,
        epsilon=0.01,
        f_star=None,
        selected_features=None,
    ):
        super().__init__(
            name="OplraRegression %d regions (f*=%s)" % (number_regions, f_star)
        )
        self.number_regions = number_regions

        # Data and parameters
        self.data = data
        self.target = target
        self.lam = lam
        self.U1 = 1.5
        self.U2 = sum(self.target)
        self.epsilon = epsilon

        # Indices of the model
        self.s = Set(initialize=data.index.tolist())
        self.r = Set(initialize=range(self.number_regions))
        if self.number_regions == 1 or selected_features is None:
            self.f = Set(initialize=self.data.columns, ordered=True)
        else:
            self.f = Set(initialize=selected_features, ordered=True)

        # Variables common to both FEATURE_SELECTION and PIECEWISE models
        self.W = Var(self.r, self.f, initialize=0)

        self.absW = Var(self.r, self.f, domain=NonNegativeReals, initialize=0)
        self.B = Var(self.r, initialize=0)
        self.Pred = Var(self.r, self.s, initialize=0)
        self.D = Var(self.s, domain=NonNegativeReals, initialize=0)
        self.F = Var(self.r, self.s, domain=Binary, initialize=0)
        self.z = Var(domain=NonNegativeReals, initialize=0)
        self.mae = Var(domain=NonNegativeReals, initialize=0)
        self.reg = Var(domain=NonNegativeReals, initialize=0)

        if self.number_regions > 1 and f_star is None:
            raise ValueError("f_star is invalid.")

        self.fStar = f_star

        # Specific variables and constraints according to model (number of regions)
        if self.number_regions == 1:
            self.sf = Var(self.f, domain=Binary, initialize=0)
            # Old equations used for explicit feature selection
            # self.number_features    = Constraint(rule=OplraPyomoModel.number_features_rule)
            # self.upper_bound_coeff  = Constraint(self.r, self.f, rule=OplraPyomoModel.upper_bound_coeff_rule)
            # self.lower_bound_coeff  = Constraint(self.r, self.f, rule=OplraPyomoModel.lower_bound_coeff_rule)
        else:
            self.isSimpleLinear = False
            self.fStar = self.fStar
            self.X = Var(self.r, bounds=(0.0, 1.0), initialize=0)
            self.rr = Set(within=self.r)
            self.breakpoint_order = Constraint(
                self.r, rule=OplraPyomoModel.breakpoint_order_rule
            )
            self.region_divider1 = Constraint(
                self.r, self.s, rule=OplraPyomoModel.region_divider1_rule
            )
            self.region_divider2 = Constraint(
                self.r, self.s, rule=OplraPyomoModel.region_divider2_rule
            )

        # Common constraints
        self.sample_to_region = Constraint(
            self.s, rule=OplraPyomoModel.sample_to_region_rule
        )
        self.prediction = Constraint(
            self.r, self.s, rule=OplraPyomoModel.prediction_rule
        )
        self.abs_error1 = Constraint(
            self.r, self.s, rule=OplraPyomoModel.abs_error1_rule
        )
        self.abs_error2 = Constraint(
            self.r, self.s, rule=OplraPyomoModel.abs_error2_rule
        )
        self.abs_coeff1 = Constraint(
            self.r, self.f, rule=OplraPyomoModel.abs_coeff1_rule
        )
        self.abs_coeff2 = Constraint(
            self.r, self.f, rule=OplraPyomoModel.abs_coeff2_rule
        )
        self.mae_eqn = Constraint(rule=OplraPyomoModel.mae_rule)
        self.reg_eqn = Constraint(rule=OplraPyomoModel.regularisation_rule)
        self.z_eqn = Constraint(rule=OplraPyomoModel.z_rule)
        self.obj = Objective(rule=OplraPyomoModel.objective_rule, sense=minimize)

    # ---- UTIL FUNCTIONS ---- #
    def get_selected_features(self):
        fs = []
        for r in self.r:
            for f in self.f:
                if self.W[r, f].value != 0 and f not in fs:
                    fs.append(f)
        return fs

    def get_breakpoints(self):
        breakpoints = [self.X[r].value for r in self.r]
        del breakpoints[-1]
        return breakpoints

    def get_regions(self, f_star_series):
        samples = range(len(f_star_series))

        if type(f_star_series) == pd.Series:
            f_star_series.index = samples
        breakpoints = self.get_breakpoints()
        regions = list(self.r)
        lo_r = {r: breakpoints[r - 1] if r != 0 else 0.0 for r in regions}
        up_r = {r: breakpoints[r] if r != len(regions) - 1 else 1 for r in regions}
        ts_distance = [
            [
                max(0, f_star_series[s] - up_r[r], lo_r[r] - f_star_series[s])
                for r in regions
            ]
            for s in samples
        ]
        ts_region = [ts_distance[s].index(min(ts_distance[s])) for s in samples]
        return np.array(ts_region)

    def get_coefficients(self):
        # TODO (improvement): use SparseCoefMixin sparsify
        return np.vstack(
            [self.W[(r, f)].value if f in self.f else 0 for f in self.data.columns]
            for r in self.r
        )

    def get_intercepts(self):
        return np.array([B.value for B in list(self.B.values())])

    # ----- EQUATIONS ---- #
    def breakpoint_order_rule(model, r):
        if r == model.number_regions - 1:
            return Constraint.Skip
        else:
            return model.X[r] + model.epsilon <= model.X[r + 1]

    def region_divider1_rule(model, r, s):
        if r == model.number_regions - 1:
            return Constraint.Skip
        else:
            return (
                model.X[r] - model.epsilon + model.U1 * (1 - model.F[r, s])
            ) >= model.data.loc[s, model.fStar]

    def region_divider2_rule(model, r, s):
        if r == 0:
            return Constraint.Skip
        else:
            return (
                model.X[r - 1] + model.epsilon - model.U1 * (1 - model.F[r, s])
            ) <= model.data.loc[s, model.fStar]

    def sample_to_region_rule(model, s):
        return sum(model.F[rr, s] for rr in model.r) == 1

    def prediction_rule(model, r, s):
        return model.Pred[r, s] == (
            sum(model.data.loc[s, ff] * model.W[r, ff] for ff in model.f) + model.B[r]
        )

    def abs_error1_rule(model, r, s):
        return model.D[s] >= (
            model.target[s] - model.Pred[r, s] - model.U2 * (1 - model.F[r, s])
        )

    def abs_error2_rule(model, r, s):
        return model.D[s] >= (
            model.Pred[r, s] - model.target[s] - model.U2 * (1 - model.F[r, s])
        )

    def abs_coeff1_rule(model, r, f):
        return model.absW[r, f] >= model.W[r, f]

    def abs_coeff2_rule(model, r, f):
        return model.absW[r, f] >= -model.W[r, f]

    def mae_rule(model):
        return model.mae == sum(model.D[ss] for ss in model.s) / len(model.s)

    def regularisation_rule(model):
        return model.reg == model.lam * sum(
            model.absW[r, f] for r in model.r for f in model.f
        )

    def z_rule(model):
        return model.z == model.mae + model.reg

    def objective_rule(model):
        return model.z

    # ----- FEATURE SELECTION EQUATIONS ---- #
    def number_features_rule(model):
        return sum(model.sf[f] for f in model.f) <= model.n_max

    def upper_bound_coeff_rule(model, r, f):
        return model.W[r, f] <= model.U2 * model.sf[f]

    def lower_bound_coeff_rule(model, r, f):
        return model.W[r, f] >= -model.U2 * model.sf[f]
