import pandas as pd
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize
import time
import pathlib
import datetime

from ...Performance import NfeloGrader
from .RecordSchema import FEATURES
from .RecordSchema import extract_performance
from .RecordSchema import RUNTIME_LOG_COLUMNS


class NfeloOptimizerBase():
    '''
    Optimizes the nfelo model
    '''
    ## set available features, ranges, and best guesses up front ##
    available_features = {
        ## format is ##
        ## 'feature' : {'best guess', 'min allowed', 'max_allowed}
        ## core Elo params ##
        'k' : {'bg':13.25, 'min':5, 'max':20},
        'z' : {'bg':400, 'min':200, 'max':600},
        'b' : {'bg':6.75, 'min':3, 'max':10},
        ## off season regression ##
        'reversion' : {'bg':0.15, 'min':0.0, 'max':1.0},
        'dvoa_weight' : {'bg':0.35, 'min':0.15, 'max':0.5},
        'wt_ratings_weight' : {'bg':0.15, 'min':0.05, 'max':0.5},
        ## score assessment ##
        'margin_weight' : {'bg':0.6, 'min':0.1, 'max':1.0},
        'pff_weight' : {'bg':0.22, 'min':0.1, 'max':1.0},
        'wepa_weight' : {'bg':0.18, 'min':0.1, 'max':1.0},
        ## shift modifiers ##
        'market_resist_factor' : {'bg':1.5, 'min':1.15, 'max':10},
        ## market reversions ##
        'market_regression' : {'bg':.8, 'min':0, 'max':.9},
        'se_span' : {'bg':4, 'min':2, 'max':16},
        'rmse_base' : {'bg':3.5, 'min':2, 'max':10},
        'spread_delta_base' : {'bg':1.5, 'min':1.1, 'max':5},
        'hook_certainty' : {'bg': -0.25, 'min':-0.5, 'max':0},
        'long_line_inflator' : {'bg':0.25, 'min':0, 'max':.75},
        'min_mr' : {'bg':0, 'min':0, 'max':0.5}
    }

    ## set of objective functions allowed ##
    available_obj_functions = {
        'nfelo_brier' : {
            'model' : 'nfelo_unregressed',
            'metric' : 'brier',
            'scale' : 10000,
            'direction' : 'pos'
        },
        'nfelo_brier_adj' : {
            'model' : 'nfelo_unregressed',
            'metric' : 'brier_adj',
            'scale' : 10000,
            'direction' : 'pos'
        },
        'nfelo_brier_close' : {
            'model' : 'nfelo_close',
            'metric' : 'brier_ats_adj',
            'scale' : 10000,
            'direction' : 'pos'
        },
        'nfelo_brier_close_ats' : {
            'model' : 'nfelo_close',
            'metric' : 'ats_be',
            'scale' : 0.10,
            'direction' : 'pos'
        }
    }

    def __init__(self,
            ## meta ##
            opti_tag,
            ## model ##
            nfelo_model, features, objective,
            ## optimizer params ##
            bg_overrides={},
            best_guesses=None, bound=(0,1),
            tol=0.000001, step=0.00001, method='SLSQP',
            results_dir=None,
        ):
        self.opti_tag = opti_tag
        self.nfelo_model = nfelo_model
        self.model_name = self.available_obj_functions[objective]['model']
        self.features = features
        self.objective = objective
        ## opti params ##
        self.bg_overrides = bg_overrides
        self.best_guesses = self.gen_best_guesses()
        self.bounds = tuple((0, 1) for _ in range(len(features))) ## all features are normalized
        self.tol = tol
        self.step = step
        self.method = method
        ## season filter is passed to the grader; None = grade every season ##
        self.season_filter = None
        ## test_season_filter, if set, triggers an additional per-save eval ##
        ## on the test seasons appended to {opti_tag}-{opti_date}_test.csv ##
        self.test_season_filter = None
        ## hop_number is tagged by RandomStarts before each optimize() call; None standalone ##
        self.hop_number = None
        ## in optimization params ##
        ## these allow the optimizer to output data as the model runs ##
        ## (saves each new best within obj_func; the original mid_opti_output pattern) ##
        self.total_runs = 0
        self.best_val = 0
        ## run_id is built per-save inside mid_opti_output (hop_number.total_runs) ##
        self.run_id = None
        ## post optimization vars ##
        self.opti_vals = []
        self.opti_seconds = 0
        self.opti_rec = {}
        ## init date captured once so multi-day runs land in one file ##
        self.opti_date = datetime.datetime.now().strftime("%Y-%m-%d")
        ## per-run output directory; defaults to Optimizer/results/ ##
        if results_dir is None:
            results_dir = pathlib.Path(__file__).parent.parent.resolve() / 'results'
        self.results_dir = pathlib.Path(results_dir)

    def _results_path(self, filename):
        '''
        Path to a results artifact inside results_dir.
        '''
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return str(self.results_dir / filename)

    def _runtime_log_path(self):
        '''
        Path to the per-eval runtime CSV for this optimization tag + date.
        '''
        return self._results_path('{0}-{1}_runtime.csv'.format(
            self.opti_tag,
            self.opti_date,
        ))

    def _log_eval_runtime(self, eval_seconds, minimized_obj):
        '''
        Appends one row per objective-function eval to the runtime CSV.
        '''
        row = {
            'optimization_type': self.opti_tag,
            'opti_date': self.opti_date,
            'hop_number': self.hop_number,
            'eval_number': self.total_runs,
            'completed_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eval_seconds': eval_seconds,
            'objective': self.objective,
            'minimized_obj': minimized_obj,
            'achieved_value': self.revert_obj(minimized_obj),
        }
        log_loc = self._runtime_log_path()
        new = pd.DataFrame([row], columns=RUNTIME_LOG_COLUMNS)
        header = not pathlib.Path(log_loc).exists()
        new.to_csv(log_loc, mode='a', header=header, index=False)


    ## NORMALIZATION FUNCTIONS ##
    ## Sets all nfelo features to be 0-1 based on allowed ranges ##
    ## to improve optimization performance ##

    def normalize_value(self, val, feature_name):
        '''
        Normalizes a value based on the allowed range
        '''
        ## get feature range
        feature_info = self.available_features[feature_name]
        min_val = feature_info['min']
        max_val = feature_info['max']
        ## normalize ##
        normalized = (val - min_val) / (max_val - min_val)
        ## return ##
        return normalized

    def denormalize_value(self, val, feaure_name):
        '''
        Denormalize a value to bring it back to its original scale
        '''
        feature_info = self.available_features[feaure_name]
        min_val = feature_info['min']
        max_val = feature_info['max']
        ## denormalize ##
        denormalized = val * (max_val - min_val) + min_val
        ## return ##
        return denormalized

    def gen_best_guesses(self):
        '''
        Generate normalized best guesses based on the array of
        features passed in the init
        '''
        best_guesses = []
        for feature in self.features:
            if feature in self.bg_overrides:
                ## override if passed ##
                best_guess = self.bg_overrides[feature]
                print('     Overriding {0} with {1}'.format(feature, best_guess))
            else:
                ## otherwise get from config ##
                feature_info = self.available_features[feature]
                best_guess = feature_info['bg']
            ## normalize ##
            normalized_best_guess = self.normalize_value(best_guess, feature)
            best_guesses.append(normalized_best_guess)
        return best_guesses

    def update_params(self, x):
        '''
        Updates the nfelo model class with the new features
        '''
        ## generate a dictionary of updates for the model ##
        updates = {}
        for i, v in enumerate(x):
            updates[self.features[i]] = self.denormalize_value(
                v,self.features[i]
            )
        ## update the config ##
        self.nfelo_model.update_config(updates)

    def metric_extraction(self, grader, model_name, metric_name):
        '''
        Extracts a grade metric from a graded nfelo model
        '''
        ## init grade ##
        grade = None
        ## get initial value from the grader records ##
        for rec in grader.graded_records:
            if rec['model_name'] == model_name:
                ## update grade ##
                grade = rec[metric_name]
        ## return ##
        return grade

    def parse_grade(self, grader):
        '''
        Translates a grade from the grader into a minimizable
        objective function based on the available_obj_functions
        config
        '''
        ## init grade ##
        grade = None
        ## get obj config ##
        obj_config = self.available_obj_functions[self.objective]
        ## get initial value from the grader records ##
        grade = self.metric_extraction(
            grader=grader,
            model_name=obj_config['model'],
            metric_name=obj_config['metric']
        )
        ## transform result ##
        grade = grade / obj_config['scale']
        if obj_config['direction'] == 'pos':
            grade *= -1
        ## return ##
        return grade

    def revert_obj(self, minimized_obj):
        '''
        Reverts the minimized obj back to a metric grade
        '''
        ## get obj config ##
        obj_config = self.available_obj_functions[self.objective]
        ## transform result ##
        grade = minimized_obj * obj_config['scale']
        if obj_config['direction'] == 'pos':
            grade *= -1
        ## return ##
        return grade

    def mid_opti_output(self, obj, grader):
        '''
        Saves a stream of optimization results while the optimizer is running
        if conditions are met. Mirrors the original mid_opti_output: tracks
        the running best across obj_func evals and writes a row each time a
        new best is found (skipping the first 15 evals to avoid SLSQP's
        initial convergence noise).
        '''
        ## see if conditions are met ##
        ## update objective function info ##
        if self.total_runs == 1:
            ## if first run, set the obj to the output ##
            self.best_val = obj
        ## see if its a new best ##
        is_new_best = True if obj < self.best_val else False
        ## update the value if needed ##
        if is_new_best:
            self.best_val = obj
        ## full conditions for output ##
        if is_new_best and self.total_runs > 15:
            ## clear optimization rec ##
            self.opti_rec = {}
            ## run_id uses total_runs so each mid-run save is unique ##
            ## hyphen rather than dot so it isn't parsed as a float in CSV readers ##
            self.run_id = '{0}-{1}'.format(
                self.hop_number if self.hop_number is not None else 1,
                self.total_runs
            )
            ## populate rec ##
            self.opti_rec['optimization_type'] = self.opti_tag
            self.opti_rec['optimization_method'] = self.method
            self.opti_rec['optimization_tol'] = self.tol
            self.opti_rec['optimization_step'] = self.step
            self.opti_rec['opti_date'] = self.opti_date
            self.opti_rec['hop_number'] = self.hop_number
            self.opti_rec['run_id'] = self.run_id
            self.opti_rec['objective'] = self.objective
            ## explicit objective model + metric so the schema is unambiguous ##
            obj_config = self.available_obj_functions[self.objective]
            self.opti_rec['objective_model'] = obj_config['model']
            self.opti_rec['objective_metric'] = obj_config['metric']
            self.opti_rec['achieved_value'] = self.revert_obj(obj)
            ## populate canonical performance columns from RecordSchema ##
            self.opti_rec.update(extract_performance(grader))
            ## populate features in canonical fixed order ##
            for feature in FEATURES:
                self.opti_rec[feature] = self.nfelo_model.config.get(feature)
            ## save ##
            self.save_to_logs()
            ## snapshot market + market_open benchmarks once per split (idempotent via file check) ##
            self._snapshot_market_benchmarks(grader, 'train')
            ## if a test filter is set, also compute test metrics and append a row to _test.csv ##
            ## skinny side table: run_id + the same 12 metrics as train, joinable post-hoc ##
            if self.test_season_filter is not None:
                test_grader = NfeloGrader(self.nfelo_model.updated_file, season_filter=self.test_season_filter)
                test_rec = {'run_id': self.run_id}
                ## canonical performance columns prefixed test_ so train+test ##
                ## frames join cleanly on run_id with no column renames ##
                for k, v in extract_performance(test_grader).items():
                    test_rec['test_{0}'.format(k)] = v
                test_log_loc = self._results_path('{0}-{1}_test.csv'.format(
                    self.opti_tag,
                    self.opti_date,
                ))
                try:
                    existing = pd.read_csv(test_log_loc, index_col=0)
                except:
                    existing = None
                new = pd.DataFrame([test_rec])
                if existing is not None:
                    new = pd.concat([existing, new]).reset_index(drop=True)
                new.to_csv(test_log_loc)
                ## snapshot test split benchmarks once ##
                self._snapshot_market_benchmarks(test_grader, 'test')

    def _snapshot_market_benchmarks(self, grader, split):
        '''
        Append market + market_open grader records for `split` to
        {opti_tag}-{opti_date}_benchmarks.csv. One-time static reference
        per split (see ANALYSIS_PLAYBOOK.md). No-ops if `split` already
        present in the file.
        '''
        bench_loc = self._results_path('{0}-{1}_benchmarks.csv'.format(
            self.opti_tag,
            self.opti_date,
        ))
        try:
            existing = pd.read_csv(bench_loc, index_col=0)
            if 'split' in existing.columns and split in existing['split'].values:
                return
        except:
            existing = None
        ## scalar metrics only -- playbook: market + market_open per split ##
        benchmark_cols = [
            'model_name', 'n_games', 'brier', 'brier_per_game', 'rmse', 'su',
            'market_correl', 'brier_adj', 'brier_ats_adj',
        ]
        rows = []
        for rec in grader.graded_records:
            if rec['model_name'] in ('market', 'market_open'):
                row = {'split': split}
                for col in benchmark_cols:
                    if col == 'rmse':
                        se = rec.get('se')
                        row[col] = float(numpy.sqrt(se)) if se is not None else None
                    else:
                        row[col] = rec.get(col)
                rows.append(row)
        if not rows:
            return
        new = pd.DataFrame(rows)
        if existing is not None:
            new = pd.concat([existing, new]).reset_index(drop=True)
        new.to_csv(bench_loc)

    def obj_func(self, x):
        '''
        Objective function for the optimizer. This will:
        * Update the model
        * Rerun the model
        * Grade the model
        * Return a score to minimize
        '''
        eval_start = float(time.time())
        ## update model ##
        self.update_params(x)
        ## rerun model ##
        self.nfelo_model.run()
        ## create a grader (respects season_filter for train/test) ##
        grader = NfeloGrader(self.nfelo_model.updated_file, season_filter=self.season_filter)
        ## get the correct metric and make it minimizable
        obj = self.parse_grade(grader)
        ## update run count ##
        self.total_runs += 1
        print('Run number {0} - {1}'.format(
            self.total_runs, obj
        ))
        ## mid-run save: write a row whenever a new best is found ##
        self.mid_opti_output(obj, grader)
        self._log_eval_runtime(float(time.time()) - eval_start, obj)
        ## return ##
        return obj

    def optimize(self):
        '''
        Function that performs the optimization
        '''
        ## optimize ##
        ## reset counter ##
        self.opti_round = 0
        ## reset per-call state so new-best tracking and the >15 skip apply per-hop ##
        self.best_val = float('inf')
        self.total_runs = 0
        opti_time_start = float(time.time())
        solution = minimize(
            self.obj_func,
            self.best_guesses,
            bounds=self.bounds,
            method=self.method,
            options={
                'ftol' : self.tol,
                'eps' : self.step
            }
        )
        if not solution.success:
            print('     FAIL')
        opti_time_end = float(time.time())
        ## update properties ##
        self.opti_seconds = opti_time_end - opti_time_start

    def save_to_logs(self, file_name=None):
        '''
        Saves the optimization record to the logs
        '''
        log_loc = self._results_path(
            file_name if file_name is not None else '{0}-{1}.csv'.format(
                self.opti_tag, self.opti_date
            )
        )
        if not log_loc.endswith('.csv'):
            log_loc = '{0}.csv'.format(log_loc)
        ## load ##
        try:
            existing = pd.read_csv(log_loc, index_col=0)
        except:
            existing = None
        ## dedup: if existing file already has a row with this run_id, skip writing ##
        if existing is not None and 'run_id' in existing.columns and self.opti_rec.get('run_id') is not None:
            if self.opti_rec['run_id'] in existing['run_id'].astype(str).values:
                return
        ## gen new ##
        new = pd.DataFrame([self.opti_rec])
        ## merge if needed
        if existing is not None:
            new = pd.concat([
                existing, new
            ]).reset_index(drop=True)
        ## save ##
        new.to_csv(log_loc)
