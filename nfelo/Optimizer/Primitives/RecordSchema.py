'''
Canonical schema for optimizer output CSVs.

Models, metrics, and features are FIXED across every run so the resulting
CSVs share a single column shape regardless of which objective the run
targeted. Column order is grouped by model, then metric -- all metrics
for nfelo_unregressed first, then nfelo_open, etc.

The grader's internal model_name 'market' represents market_close; this
module aliases it to 'market_close' so the schema name is unambiguous
without touching NfeloGrader.
'''
import numpy

MODELS = [
    'nfelo_unregressed',
    'nfelo_open',
    'nfelo_close',
    'market_open',
    'market_close',
]

## schema model name -> NfeloGrader model_name ##
_GRADER_KEY = {
    'nfelo_unregressed': 'nfelo_unregressed',
    'nfelo_open':        'nfelo_open',
    'nfelo_close':       'nfelo_close',
    'market_open':       'market_open',
    'market_close':      'market',
}

METRICS = [
    'brier',
    'brier_per_game',
    'rmse',
    'su',
    'ats',
    'ats_be',
    'ats_be_play_pct',
    'market_correl',
    'brier_adj',
    'brier_ats_adj',
]

## market models use the market line as both model and market, so ATS cols ##
## are always blank -- omit them from the canonical export schema ##
_ATS_METRICS = {'ats', 'ats_be', 'ats_be_play_pct'}
_MARKET_MODELS = {'market_open', 'market_close'}

FEATURES = [
    'k', 'z', 'b', 'reversion', 'dvoa_weight',
    'wt_ratings_weight', 'margin_weight', 'pff_weight',
    'wepa_weight', 'market_resist_factor',
    'market_regression', 'se_span', 'rmse_base',
    'spread_delta_base', 'hook_certainty',
    'long_line_inflator', 'min_mr',
]

RUNTIME_LOG_COLUMNS = [
    'optimization_type',
    'opti_date',
    'hop_number',
    'eval_number',
    'completed_at',
    'eval_seconds',
    'objective',
    'minimized_obj',
    'achieved_value',
]

def model_metrics(schema_model:str) -> list:
    '''
    Metrics exported for a schema model. Market models omit ATS columns.
    '''
    if schema_model in _MARKET_MODELS:
        return [m for m in METRICS if m not in _ATS_METRICS]
    return METRICS

def performance_columns() -> list:
    '''
    Returns the canonical performance column names in fixed order
    (grouped by model, then metric in METRICS order).
    '''
    return [
        '{0}_{1}'.format(metric, model)
        for model in MODELS
        for metric in model_metrics(model)
    ]

def extract_performance(grader) -> dict:
    '''
    Pulls the canonical {metric}_{model} performance dict from a graded
    NfeloGrader. RMSE is derived as sqrt(grader's mean squared error).

    Parameters:
    * grader (NfeloGrader): an instance whose grade_models() has run

    Returns:
    * out (dict): keyed by canonical column names, in canonical order
    '''
    ## index grader records by their internal model_name ##
    by_grader_name = {rec['model_name']: rec for rec in grader.graded_records}
    out = {}
    for schema_model in MODELS:
        grader_name = _GRADER_KEY[schema_model]
        rec = by_grader_name.get(grader_name, {})
        for metric in model_metrics(schema_model):
            out['{0}_{1}'.format(metric, schema_model)] = _read_metric(rec, metric)
    return out

def _read_metric(rec:dict, metric:str) -> (float or None):
    '''
    Reads a single metric from a grader score_record. RMSE is derived
    from `se` (mean squared error); all other metrics pass through.

    Parameters:
    * rec (dict): a NfeloGraderModel.gen_score_record() output (or empty)
    * metric (str): canonical metric name from METRICS

    Returns:
    * value (float or None): the metric value, or None if missing
    '''
    if not rec: return None
    if metric == 'rmse':
        se = rec.get('se')
        if se is None: return None
        return float(numpy.sqrt(se))
    return rec.get(metric)
