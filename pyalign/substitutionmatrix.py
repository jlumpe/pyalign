import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import eig


__all__ = ['aa_symbols', 'SubstitutionMatrix', 'PamMatrix']


# List of amino acid single-letter codes
aa_symbols = list('ACDEFGHIKLMNPQRSTVWY')


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

	def score(self, s1, s2):
		try:
			return self.matrix[self.symbol_map[s1], self.symbol_map[s2]]
		except KeyError:
			return 0

	def plot(self, ax=None, colorbar=True, **kwargs):
		from matplotlib import pyplot as plt

		if ax is None:
			import pylab
			ax = pylab.gca()

		img = ax.imshow(self.matrix, interpolation='none', **kwargs)

		ax.set_xticks(range(20))
		ax.set_yticks(range(20))
		ax.set_xticklabels(map(str, self.symbols))
		ax.set_yticklabels(map(str, self.symbols))

		if colorbar:
			plt.colorbar(img)

		return img


# A Model of Evolutionary Change in Proteins, Dayhoff 1978
# Note this is a LEFT stochastic matrix, all columns sum to 1.
# Except they don't, I assume the values have been rounded in the original
# paper to the nearest 10,000th. Normalize by column sums.
pam1_matrix = np.asarray([
	[9867,3,10,17,2,21,2,6,2,4,6,9,22,8,2,35,32,18,0,2],
	[1,9973,0,0,0,0,1,1,0,0,0,0,1,0,1,5,1,2,0,3],
	[6,0,9859,53,0,6,4,1,3,0,0,42,1,6,0,5,3,1,0,0],
	[10,0,56,9865,0,4,2,3,4,1,1,7,3,35,0,4,2,2,0,1],
	[1,0,0,0,9946,1,2,8,0,6,4,1,0,0,1,2,1,0,3,28],
	[21,1,11,7,1,9935,1,0,2,1,1,12,3,3,1,21,3,5,0,0],
	[1,1,3,1,2,0,9912,0,1,1,0,18,3,20,8,1,1,1,1,4],
	[2,2,1,2,7,0,0,9872,2,9,12,3,0,1,2,1,7,33,0,1],
	[2,0,6,7,0,2,2,4,9926,1,20,25,3,12,37,8,11,1,0,1],
	[3,0,0,1,13,1,4,22,2,9947,45,3,3,6,1,1,3,15,4,2],
	[1,0,0,0,1,0,0,5,4,8,9874,0,0,2,1,1,2,4,0,0],
	[4,0,36,6,1,6,21,3,13,1,0,9822,2,4,1,20,9,1,1,4],
	[13,1,1,3,1,2,5,1,2,2,1,2,9926,8,5,12,4,2,0,0],
	[3,0,5,27,0,1,23,1,6,3,4,4,6,9876,9,2,2,1,0,0],
	[1,1,0,0,1,0,10,3,19,1,4,1,4,10,9913,6,1,1,8,0],
	[28,11,7,6,3,16,2,2,7,1,4,34,17,4,11,9840,38,2,5,2],
	[22,1,4,2,1,2,1,11,8,2,6,13,5,3,2,32,9871,9,0,2],
	[13,3,1,2,1,3,3,57,1,11,17,1,3,2,2,2,10,9901,0,2],
	[0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,1,0,0,9976,1],
	[1,3,0,1,21,0,4,1,0,1,0,3,0,0,0,1,1,1,2,9945]
], dtype=np.float64)
pam1_matrix /= pam1_matrix.sum(axis=0, keepdims=True)

# Find the stationary vector of the PAM1 matrix. This corresponds to the
# eigenvector with eigenvalue 1 (a stochastic matrix has maximum eigenvalue
# 1 due to so-and-so's theorem or whatever). Use scipy's eig() function
# instead of numpy's, as it seems to be more stable. Note that it returns a
# matrix with eigenvectors in columns, normalized under L2 norm.
_pam1_eigvals, _pam1_eigvecs = eig(pam1_matrix)
pam_stationary = _pam1_eigvecs[:, np.argmax(_pam1_eigvals)].copy()
pam_stationary /= pam_stationary.sum()


class PamMatrix(SubstitutionMatrix):
	"""
	PAM scoring matrix. Creates scoring matrix from log-odds ratio of
	PAM{n}:PAM{inf}.
	"""

	def __init__(self, n):
		self.n = n

		# Calculate matrix
		pam_n = matrix_power(pam1_matrix, n)
		matrix = np.log10(pam_n / pam_stationary[:, None])

		# Doesn't seem to be completely symmetrical, hopefully just due to
		# 	rounding errors. Fudge it a bit.
		matrix += matrix.transpose()
		matrix *= .5

		SubstitutionMatrix.__init__(self, aa_symbols, matrix.astype(np.float32))
