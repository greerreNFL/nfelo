import pandas as pd
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize, basinhopping
import time
import pathlib
import datetime

from ..Performance import NfeloGrader


class NfeloOptimizer():
    '''
    Optimizes the nfelo model
    '''
    ## set available features, ranges, and best guesses up front ##
    available_features = {
        ## format is ##
        ## 'feature' : {'best guess', 'min allowed', 'max_allowed}
        ## core Elo params ##
        'k' : {'bg':13.5, 'min':5, 'max':20},
        'z' : {'bg':400, 'min':200, 'max':600},
        'b' : {'bg':6.75, 'min':3, 'max':10},
        ## off season regression ##
        'reversion' : {'bg':0.3, 'min':.15, 'max':1},
        'dvoa_weight' : {'bg':0.35, 'min':0.15, 'max':0.5},
        'wt_ratings_weight' : {'bg':0.35, 'min':0.15, 'max':0.5},
        ## score assessment ##
        'margin_weight' : {'bg':0.6, 'min':0.1, 'max':1.0},
        'pff_weight' : {'bg':0.22, 'min':0.1, 'max':1.0},
        'wepa_weight' : {'bg':0.18, 'min':0.1, 'max':1.0},
        ## shift modifiers ##
        'market_resist_factor' : {'bg':2.5, 'min':1.25, 'max':10},
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
            'metric' : 'brier_adj',
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
            best_guesses=None, bound=(0,1), 
            tol=0.000001, step=0.00001, method='SLSQP',
            basin_hop=False
        ):
        self.opti_tag = opti_tag
        self.nfelo_model = nfelo_model
        self.model_name = self.available_obj_functions[objective]['model']
        self.features = features
        self.objective = objective
        ## opti params ##
        self.best_guesses = self.gen_best_guesses()
        self.bounds = tuple((0, 1) for _ in range(len(features))) ## all features are normalized
        self.tol = tol
        self.step = step
        self.method = method
        self.basin_hop = basin_hop
        ## post optimization vars ##
        self.opti_vals = []
        self.opti_seconds = 0
        self.opti_rec = {}

    
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
            feature_info = self.available_features[feature]
            best_guess = feature_info['bg']
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
    
    def obj_func(self, x):
        '''
        Objective function for the optimizer. This will:
        * Update the model
        * Rerun the model
        * Grade the model
        * Return a score to minimize
        '''
        ## update model ##
        self.update_params(x)
        ## rerun model ##
        self.nfelo_model.run()
        ## create a grader ##
        grader = NfeloGrader(self.nfelo_model.updated_file)
        ## get the correct metric and make it minimizable
        obj = self.parse_grade(grader)
        print(obj)
        ## return ##
        return obj

    def optimize(self):
        '''
        Function that performs the optimization
        '''
        ## optimize ##
        ## reset counter ##
        self.opti_round = 0
        opti_time_start = float(time.time())
        if self.basin_hop:
            solution = basinhopping(
                self.obj_func,
                self.best_guesses,
                minimizer_kwargs={
                    'method' : self.method,
                    'bounds' : self.bounds,
                    'options' :{
                        'ftol' : self.tol,
                        'eps' : self.step
                    }
                }
            )
        else:
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
        ## save a graded output of the final version ##
        ## make sure the model has used the most recent config ##
        self.update_params(solution.x)
        self.nfelo_model.run()
        graded = NfeloGrader(self.nfelo_model.updated_file)
        graded.print_scores()
        graded.save_scores(
            '{0}/graded.csv'.format(
                pathlib.Path(__file__).parent.resolve()
            )
        )
        ## save the optimization record
        for i, v in enumerate(solution.x):
            self.opti_vals.append(self.denormalize_value(v, self.features[i]))
        ## construct the record ##
        self.opti_rec = {}
        self.opti_rec['optimization_type'] = self.opti_tag
        self.opti_rec['opti_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
        self.opti_rec['run_time'] = self.opti_seconds
        self.opti_rec['iterations'] = solution.nit
        self.opti_rec['avg_time_per_eval'] = self.opti_seconds / solution.nit
        self.opti_rec['objective'] = self.objective
        self.opti_rec['achieved_value'] = self.revert_obj(solution.fun)
        ## add performance data ##
        self.opti_rec['brier'] = self.metric_extraction(graded, self.model_name, 'brier')
        self.opti_rec['brier_per_game'] = self.metric_extraction(graded, self.model_name, 'brier_per_game')
        self.opti_rec['brier_adj'] = self.metric_extraction(graded, self.model_name, 'brier_adj')
        self.opti_rec['su'] = self.metric_extraction(graded, self.model_name, 'su')
        self.opti_rec['ats'] = self.metric_extraction(graded, self.model_name, 'ats')
        self.opti_rec['ats_be'] = self.metric_extraction(graded, self.model_name, 'ats_be')
        self.opti_rec['ats_be_play_pct'] = self.metric_extraction(graded, self.model_name, 'ats_be_play_pct')
        self.opti_rec['market_correl'] = self.metric_extraction(graded, self.model_name, 'market_correl')
        ## final model specific metrics ##
        self.opti_rec['brier_nfelo_close'] = self.metric_extraction(graded, 'nfelo_close', 'brier')
        self.opti_rec['ats_nfelo_close'] = self.metric_extraction(graded, 'nfelo_close', 'ats')
        self.opti_rec['ats_be_nfelo_close'] = self.metric_extraction(graded, 'nfelo_close', 'ats_be')
        ## populate features ##
        for feature, v in self.available_features.items():
            self.opti_rec[feature] = self.nfelo_model.config[feature]
    
    def save_to_logs(self):
        '''
        Saves the optimization record to the logs
        '''
        log_loc = '{0}/opti_logs.csv'.format(
            pathlib.Path(__file__).parent.resolve()
        )
        ## load ##
        try:
            existing = pd.read_csv(log_loc, index_col=0)
        except:
            existing = None
        ## gen new ##
        new = pd.DataFrame([self.opti_rec])
        ## merge if needed
        if existing is not None:
            new = pd.concat([
                existing, new
            ]).reset_index(drop=True)
        ## save ##
        new.to_csv(log_loc)