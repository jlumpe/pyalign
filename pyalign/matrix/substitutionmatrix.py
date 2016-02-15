# List of amino acid single-letter codes
aa_symbols = list('ACDEFGHIKLMNPQRSTVWY')

# Extended set of symbols, including B, Z, and X ambiguitiy codes and stop
# codon (*)
aa_symbols_ext = list('ARNDCQEGHILKMFPSTWYVBZX*')


class SubstitutionMatrix(object):
	"""
	Represents a substitution/scoring matrix for alignments.
	"""

	def __init__(self, symbols, matrix):
		assert matrix.shape == (len(symbols), len(symbols))
		self.symbols = list(symbols)
		self.matrix = matrix
		self.symbol_map = {s: i for i, s in enumerate(symbols)}

	def __contains__(self, symbol):
		return symbol in self.symbol_map

	def __getnewargs__(self):
		"""For the pickling protocol"""
		return self.symbols, self.matrix

	def score(self, s1, s2):
		try:
			return self.matrix[self.symbol_map[s1], self.symbol_map[s2]]
		except KeyError:
			return 0

	def plot(self, ax=None, colorbar=True, **kwargs):

		if ax is None:
			from matplotlib import pyplot as plt
			ax = plt.gca()

		img = ax.imshow(self.matrix, interpolation='none', **kwargs)

		ax.set_xticks(range(20))
		ax.set_yticks(range(20))
		ax.set_xticklabels(map(str, self.symbols))
		ax.set_yticklabels(map(str, self.symbols))

		if colorbar:
			ax.figure.colorbar(img)

		return img
