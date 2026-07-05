'''
Builds scipy SLSQP constraint dicts from registry specs.
'''
from dataclasses import dataclass


@dataclass
class ConstraintSpec():
    '''
    Entry format for NfeloOptimizerBase.available_constraints.
    '''
    members: list ## feature names summed by this constraint
    limit: float ## target sum (<= limit for ineq, == limit for eq)
    kind: str = 'ineq' ## 'ineq' (sum <= limit) or 'eq' (sum == limit)


class Constraint():
    '''
    Converts a ConstraintSpec + feature bounds into the dict
    scipy.optimize.minimize expects for SLSQP.

    Parameters:
    * spec (ConstraintSpec): registry entry
    * feature_bounds (dict): {feature: {'min', 'max', 'index'}} for active members
    '''

    def __init__(self,
            spec,
            feature_bounds,
        ):
        self.spec = spec
        self.feature_bounds = feature_bounds

    def to_scipy(self):
        '''
        Returns:
        * constraint (dict): {'type': 'ineq'|'eq', 'fun': callable}
        '''
        limit = float(self.spec.limit)
        bounds = self.feature_bounds
        kind = self.spec.kind

        ## construct function for scipy
        def fun(x):
            total = 0.0
            for feat, info in bounds.items():
                val = x[info['index']]
                total += val * (info['max'] - info['min']) + info['min']
            if kind == 'eq':
                return total - limit
            return limit - total

        ## return in scipy format
        return {'type': kind, 'fun': fun}
