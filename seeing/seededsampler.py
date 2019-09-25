from torch.utils.data.sampler import Sampler
import numpy

class SeededRandomSampler(Sampler):
    """Samples elements randomly, using a fixed seed."""
    def __init__(self, data_source, replacement=False, num_samples=None,
            seed=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.rng = numpy.random.RandomState(seed)

        if self.num_samples is not None and replacement is False:
            raise ValueError(
               "With replacement=False, num_samples should not be specified, "
               "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(self.rng.randint(high=n,
                size=(self.num_samples,), dtype=numpy.int64).tolist())
        return iter(self.rng.permutation(n)[:self.num_samples].tolist())

    def __len__(self):
        return len(self.data_source)
