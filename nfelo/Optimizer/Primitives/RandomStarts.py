import numpy


class RandomStarts():
    '''
    Composes a NfeloOptimizerBase and runs N independent local minimizations,
    each starting from a fresh uniform-random point in normalized [0,1] space
    over ONLY the optimized features. No chain state, no perturbation, no
    Metropolis acceptance -- each hop is an independent random restart.

    Each hop calls base.optimize() (which auto-saves a row tagged with the
    hop_number set on the base before the call).

    RandomStarts does not save anything itself -- base owns the save.
    '''

    def __init__(self,
            ## composed primitive ##
            base,
            ## hop loop params ##
            niter=30,
        ):
        self.base = base
        self.niter = niter
        self.rng = numpy.random.default_rng()

    def optimize(self):
        '''
        Run the random-restart loop.

        For each hop in 0..niter-1, generate a fresh uniform-random starting
        point in normalized [0,1] space for each OPTIMIZED feature (not the
        full available_features set), set it as base.best_guesses, tag the
        hop number on base, and call base.optimize() which auto-saves a row.
        '''
        for hop in range(1, self.niter + 1):
            ## fresh random start in normalized space, one entry per optimized feature ##
            x_random = self.rng.uniform(0.0, 1.0, size=len(self.base.features))
            self.base.best_guesses = x_random.tolist()
            self.base.hop_number = hop
            self.base.opti_vals = []
            self.base.optimize()

    def save_to_logs(self, file_name=None):
        '''
        Pass-through to base. RandomStarts owns no records of its own.
        '''
        return self.base.save_to_logs(file_name=file_name)
