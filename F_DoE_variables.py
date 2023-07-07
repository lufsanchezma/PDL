

class DoE:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_inputs = None
        self.n_inputs_aux = None
        self.limits = None
        self.n_outputs = None

        self.variation = None

        self.n_samples_main = None
        self.method_main = None
        self.n_samples_secondary = None
        self.method_secondary = None
        self.normal_boundaries_secondary = None
        self.normal_boundaries_main = None
