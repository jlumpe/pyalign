"""
Defines the SubstitutionMatrix class and basic functions for creating them.
"""

import numpy as np


__all__ = ['SubstitutionMatrix', 'aa_symbols', 'aa_symbols_ext',
	'identity_matrix']


# List of amino acid single-letter codes
aa_symbols = list('ACDEFGHIKLMNPQRSTVWY')

# Extended set of symbols, including B, Z, and X ambiguitiy codes and stop
# codon (*)
aa_symbols_ext = list('ARNDCQEGHILKMFPSTWYVBZX*')


def matrix_colormap(*matrices, **kwargs):
	"""Creates a matplotlib color map for one or more substitution matrices.

	Matices passed as *args in numpy.ndarray format.
	"""
	from matplotlib import pyplot as plt

	base_map = kwargs.get('base_map', plt.get_cmap('RdYlBu'))

	vmin = kwargs.get('vmin', min(np.min(m) for m in matrices))
	vmax = kwargs.get('vmax', max(np.max(m) for m in matrices))


class SubstitutionMatrix(object):
	"""
	Represents a substitution/scoring matrix for scoring alignments.
	"""

	def __init__(self, symbols, values, missing_score=0.):
		"""
		Args:
			symbols: sequence of hashables. Alphabet of strings in alignments
				this matrix will be used to score. Usually characters.
			values: numpy.ndarray. Symmetric 2D matrix of match scores between
				each pair of symbols. Order of rows/columns must match order
				of symbols.
			missing_score: float. Score when matching a symbol not contained
				in the list of symbols.
		"""
		assert values.shape == (len(symbols), len(symbols))
		self.symbols = list(symbols)
		self.values = values
		self.symbol_map = {s: i for i, s in enumerate(symbols)}
		self.missing_score = missing_score

	def __contains__(self, symbol):
		"""Check if a symbol is used in the matrix"""
		return symbol in self.symbol_map

	def __getnewargs__(self):
		"""For the pickling protocol"""
		return self.symbols, self.values, self.missing_score

	def score(self, s1, s2):
		"""Gets score for match between two symbols"""
		try:
			return float(self.values[self.symbol_map[s1], self.symbol_map[s2]])
		except KeyError:
			return self.missing_score

	def scale(self, s):
		"""Returns a copy of the matrix with values scaled by parameter"""
		return SubstitutionMatrix(self.symbols, self.values * s,
			self.missing_score * s)

	def plot(self, ax=None, colorbar=True, **kwargs):
		"""Plot using matplotlib (WIP)"""

		if ax is None:
			from matplotlib import pyplot as plt
			ax = plt.gca()

		img = ax.imshow(self.values, interpolation='none', **kwargs)

		ax.set_xticks(range(len(self.symbols)))
		ax.set_yticks(range(len(self.symbols)))
		ax.set_xticklabels(map(str, self.symbols))
		ax.set_yticklabels(map(str, self.symbols))

		if colorbar:
			ax.figure.colorbar(img)

		return img


def identity_matrix(symbols, id_score=1., diff_score=0., **kwargs):
	"""
	Creates a SubstitutionMatrix where matches are scored on identity alone.

	Args:
		symbols: sequence of hashables. Symbols for matrix.
		id_score: float. Score for an identical match.
		diff_score: float. Score for a non-identical match.

	**kwargs:
		missing_score: float. Score for matching a symbol not in the list.
			Defaults to diff_score.
		ambiguous: sequence of bool. If given, symbols with indices matching
			the true values in this sequence are considered ambiguous and
			are not considered identical with themselves.

	Returns:
		SubstitutionMatrix
	"""
	# Keyword args
	ambigous = kwargs.get('ambigous', None)
	missing_score = kwargs.get('missing_score', diff_score)

	if len(kwargs):
		raise TypeError('Unknown keyword argument {}'.format(
			repr(next(iter(kwargs)))))

	# Create matrix values
	n = len(symbols)
	values = np.full((n, n), diff_score, dtype=np.float32)
	np.fill_diagonal(values, id_score)

	# Remove identity scores for ambiguous symbols if needed
	if ambiguous is not None:
		for i, a in enumerate(ambigous):
			if a:
				values[i, i] = diff_score

	# Create matrix
	return SubstitutionMatrix(symbols, values, missing_score=missing_score)
