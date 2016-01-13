"""Functions for pairwise global alignment"""

import numpy as np

import _pwalign
from ..pairwisealignment import PairwiseAlignment


def pw_global_align(seqs, sub_matrix, gap_penalty):
	"""
	Find pairwise global alignment of two sequences through Hirschberg's
	algorithm.

	Args:
		seqs: 2-tuple. Sequences of (byte) characters.
		sub_matrix: SubstitutionMatrix.
		gap_penalty: int|tuple. Either a 2-tuple of ints representing gap open
			and extend penalties, or a single int for both.

	Returns:
		PairwiseAlignment
	"""

	# Parse arguments
	seq1, seq2 = seqs

	try:
		gap_open_penalty, gap_extend_penalty = gap_penalty
	except ValueError:
		gap_open_penalty = gap_penalty
		gap_extend_penalty = gap_penalty

	# Map symbols to matrix indices
	symbol_map = {s: (i + 1) for i, s in enumerate(sub_matrix.symbols)}
	ns = len(symbol_map)

	# Create matrix, zero index corresponds to symbol not in passed matrix
	matrix = np.zeros((256, 256), dtype=np.float32)
	matrix[1:(ns + 1), 1:(ns + 1)] = sub_matrix.matrix

	# Convert to numpy unit8 arrays
	seq1_array = _seq_to_array(seq1, symbol_map)
	seq2_array = _seq_to_array(seq2, symbol_map)

	# Run native algorithm to do the actual work
	gaps = _pwalign.pwalign(seq1_array, seq2_array, matrix,
		gap_open_penalty, gap_extend_penalty)

	# Return PairwiseAlignment object
	return PairwiseAlignment((seq1, seq2), gaps.astype(np.bool))


def _seq_to_array(seq, symbol_map):
	"""
	Converts a sequence to a numpy unit8 array according to alphabet position
	"""
	array = np.ndarray(len(seq), dtype=np.uint8)
	for i, s in enumerate(seq):
		array[i] = symbol_map.get(s, 0)
	return array
