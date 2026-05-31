from .Primitives.NfeloOptimizerBase import NfeloOptimizerBase
from .Primitives.RandomStarts import RandomStarts


class NfeloOptimizer():
    '''
    Orchestrator over the optimization primitives. Keeps the existing external
    API and composes:
        * NfeloOptimizerBase  -- single SLSQP local optimization that saves on each new best
        * RandomStarts        -- runs many base.optimize() calls if random_starts=True
    Train/test split is handled here: when test_seasons is non-empty the base's
    grader is filtered to train seasons during optimization, and the base writes
    an extra row to {name}_test.csv on every new best (using test_season_filter).
    '''

    def __init__(self,
            ## meta ##
            opti_tag,
            ## model ##
            nfelo_model, features, objective,
            ## optimizer params ##
            bg_overrides={},
            best_guesses=None, bound=(0,1),
            tol=0.000001, step=0.00001, method='SLSQP',
            random_starts=False,
            niter=30,
            ## test/train split ##
            test_seasons=None,
        ):
        ## build the base primitive that runs one SLSQP per call ##
        self.base = NfeloOptimizerBase(
            opti_tag,
            nfelo_model, features, objective,
            bg_overrides=bg_overrides,
            best_guesses=best_guesses, bound=bound,
            tol=tol, step=step, method=method,
        )
        ## wrap with random starts if requested ##
        if random_starts:
            self.strategy = RandomStarts(self.base, niter=niter)
        else:
            self.strategy = self.base
        ## train/test state ##
        self.test_seasons = test_seasons
        ## expose nfelo_model so existing callers (Development/optimization.py) keep working ##
        self.nfelo_model = nfelo_model

    def compute_train_seasons(self):
        '''
        Train seasons = all played seasons in the data minus test_seasons.
        '''
        played = self.base.nfelo_model.data.current_file[
            self.base.nfelo_model.data.current_file['home_margin'].notna()
        ]
        all_seasons = sorted(played['season'].unique().tolist())
        return [s for s in all_seasons if s not in self.test_seasons]

    def optimize(self):
        '''
        Run the optimization. If test_seasons is non-empty, the base grades on
        train seasons during the optimization, and writes a parallel test row
        to {name}_test.csv on every new best (handled inside mid_opti_output).

        Critically, the optimizer will run the model across all seasons to preserve
        the directional nature of an Elo model. The train/test split occurs at the grading
        level in which the optimizer will only receive metrics from test seasons.
        '''
        ## set train + test filters if a holdout was specified ##
        if self.test_seasons:
            train_seasons = self.compute_train_seasons()
            self.base.season_filter = train_seasons
            self.base.test_season_filter = self.test_seasons
            print('Train/test split: train={0} seasons, test={1}'.format(
                len(train_seasons), self.test_seasons
            ))
        ## delegate to strategy (base or random starts) ##
        self.strategy.optimize()

    def save_to_logs(self, file_name=None):
        '''
        Pass-through to base.save_to_logs.
        '''
        return self.base.save_to_logs(file_name=file_name)
