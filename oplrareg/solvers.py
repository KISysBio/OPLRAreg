from pyomo.environ import SolverFactory
from abc import ABCMeta, abstractmethod


def get_solver_definition(solver_name, **kwargs):
    if solver_name.lower() == "cplex":
        return CplexDefinition(**kwargs)
    elif solver_name.lower() == "glpk":
        return GlpkDefinition(**kwargs)
    else:
        raise ValueError('Solver %s is not supported' % solver_name)


class SolverDefinition(metaclass=ABCMeta):
    def __init__(self, solver_name, time_limit=120, threads=4, mipgap=0.01):
        self.timeLimit = time_limit
        self.threads = threads
        self.mipgap = mipgap
        self.solver = SolverFactory(solver_name)
        self.update()

    @abstractmethod
    def update(self):
        pass

    def solve(self, pyomo_model, verbose=False):
        if verbose:
            results = self.solver.solve(pyomo_model, keepfiles=True,
                                        tee=True, symbolic_solver_labels=True)
        else:
            results = self.solver.solve(pyomo_model, keepfiles=False,
                                        tee=False, symbolic_solver_labels=True)
        pyomo_model.solutions.load_from(results)
        return results.solver.termination_condition


class CplexDefinition(SolverDefinition):
    def __init__(self, **kwargs):
        super().__init__("cplex", **kwargs)

    def update(self):
        self.solver.options['timelimit'] = self.timeLimit
        self.solver.options['threads'] = self.threads
        self.solver.options['mipgap'] = self.mipgap


class GlpkDefinition(SolverDefinition):
    def __init__(self, **kwargs):
        super().__init__("glpk", **kwargs)

    def update(self):
        self.solver.options['tmlim'] = self.timeLimit
        self.solver.options['mipgap'] = self.mipgap
