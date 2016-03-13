"""
Contains code for generating PAM matrices, according to "A Model of
Evolutionary Change in Proteins" (Dayhoff 1978).
"""

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import eig

from .substitutionmatrix import SubstitutionMatrix, aa_symbols


# Mutability matrix from the original paper in 1/10,000ths, re-ordered so that
# the AA codes are in alphabetical order (as defined in this sub-package).
# Note this is a LEFT stochastic matrix, all columns sum to 1.
# (Except they don't, I assume the values have been rounded in the original
# paper to the nearest 10,000th. Normalize by column sums.)
dayhoff_matrix = np.asarray([
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
dayhoff_matrix /= dayhoff_matrix.sum(axis=0, keepdims=True)

# Find the stationary vector of the Dayhoff matrix. This corresponds to the
# eigenvector with eigenvalue 1 (a stochastic matrix has maximum eigenvalue
# 1 due to so-and-so's theorem or whatever). Use scipy's eig() function
# instead of numpy's, as it seems to be more stable. Note that it returns a
# matrix with eigenvectors in columns, normalized under L2 norm.
dayhoff_eigvals, dayhoff_eigvecs = eig(dayhoff_matrix)
assert not np.any(np.iscomplex(dayhoff_eigvecs))
assert not np.any(np.iscomplex(dayhoff_eigvals))
dayhoff_stationary = dayhoff_eigvecs[:, np.argmax(dayhoff_eigvals)]\
	.astype(np.float64)
dayhoff_stationary /= dayhoff_stationary.sum()


def pam_matrix(d, scale=float(np.log(2)/2), as_ints=False):
	"""Creates PAM scoring matrix.

	Values calculated from (natural) log-odds ratio of PAM{d}:PAM{inf}. The
	output will be this matrix *divided* by the scale parameter (this seems
	to be the convention, but I am not sure why).

	Args:
		d: int. Mutation distance (number of time steps in the Markov chain
			model).
		scale: float. Units of returned matrix, relative to "natural" units
			of the log-odds ration. Returned matrix will be the log-odds
			values *divided* by the scale. Defaults to ln(2)/2 (half-bit
			units), because that's what I've seen everywhere.
		as_ints: bool. If true, entries will be rounded to the nearest integer.
	"""
	# Calculate matrix
	dayhoff_n = matrix_power(dayhoff_matrix, d)
	matrix = np.log(dayhoff_n / dayhoff_stationary[:, None]) / scale

	# Doesn't seem to be completely symmetrical, hopefully just due to
	# floating-point errors. Fudge it a bit.
	matrix += matrix.transpose()
	matrix *= .5

	# Round if requested
	if as_ints:
		matrix = np.round(matrix)

	return SubstitutionMatrix(aa_symbols, matrix.astype(np.float32))



def map_symbols(matrix, old_symbols, new_symbols):
	map_ = tuple(old_symbols.index(s) for s in new_symbols)
	return matrix[map_, :][:, map_]


def compare_pam():
	"""Loads all NCBI PAM matrices and compare's against pyalign's method"""
	from pyalign.matrix import matrix_attrs, load_matrix

	matrices = []

	# Check all matrices
	for name, attrs in matrix_attrs.iteritems():
		# Find PAM matrices
		try:
			if attrs['type'] != 'PAM':
				continue
			dist = attrs['dist']
			scale = attrs['scale']
		except KeyError:
			continue

		# Load NCBI matrix
		ncbi_matrix = load_matrix(name)


		# Load NCBI matrix
		ncbi_matrix = load_matrix(name)

		# Create pyalign matrix
		pyalign_matrix = pam_matrix(dist, scale=scale)

		# Get values in same order
		pyalign_values = pyalign_matrix.values
		index_map = [ncbi_matrix.symbols.index(s) for s in pyalign_matrix.symbols]
		ncbi_values = ncbi_matrix.values[index_map, :][:, index_map]

		matrices.append(dict(
			name=name,
			dist=dist,
			scale=scale,
			ncbi_values=ncbi_values,
			pyalign_values=pyalign_values
		))

	return matrices


def test_pam():
	"""Test PAM matrices calculated by my method vs those from NCBI"""
	import pandas as pd

	# Rows of table to return
	rows = []
	names = []

	# Check all matrices
	for comp in compare_pam():

		# Get difference
		diff = np.abs(comp['ncbi_values'] - comp['pyalign_values'])

		# Add row
		names.append(comp['name'])
		rows.append(dict(
			dist=comp['dist'],
			scale=comp['scale'],
			meandiff=np.mean(diff),
			mindiff=np.min(diff),
			maxdiff=np.max(diff),
			nround=np.sum(diff > .5)
		))

	table = pd.DataFrame(rows, index=names, columns=['dist', 'scale',
		'meandiff', 'mindiff', 'maxdiff', 'nround'])

	table.sort_values(['dist', 'scale'], inplace=True)

	return table

