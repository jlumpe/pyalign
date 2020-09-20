import sys

import numpy as np


class PairwiseAlignment(object):
	'''Pairwise alignment between two sequences'''

	def __init__(self, seqs, gaps):
		self.seqs = tuple(seqs)
		self.gaps = gaps
		self.indices = np.cumsum(~self.gaps, axis=1) - 1
		self.indices[self.gaps] = 0

	def __len__(self):
		return self.gaps.shape[1]

	def char_at(self, seq, pos):
		'''
		Get character of sequence at specific position. Returns None if gap.
		'''
		if self.gaps[seq, pos]:
			return None
		else:
			return self.seqs[seq][self.indices[seq, pos]]

	def gap_at(self, pos):
		'''
		Returns true if there is a gap at the specified position in the
		alignment.
		'''
		return bool(np.sum(self.gaps[:, pos]))

	def match_at(self, pos):
		'''
		Returns true if both sequences match at the specified position in the
		alignment.
		'''
		return not self.gap_at(pos) and \
			self.char_at(0, pos) == self.char_at(1, pos)

	def get_identity(self, method='mismatch'):
		'''
		Calculates fractional identity of the alignments, according to one
		of several methods:
			'mismatch' - identities / (identities + mismatches)
			'columns' - identities / columns
			0 - identities / len(seqs[0])
			1 - identities / len(seqs[1])
			'shorter' - same as 0 if len(seqs[0]) < len(seqs[1]) else 1
			'total' - total identities only
		'''
		# Calculate total number of identities
		identities = sum(self.match_at(i) for i in range(len(self)))

		# Return count only if requested
		if method == 'identities':
			return identities

		# Pick shorter sequence
		if method == 'shorter':
			method = 0 if len(self.seqs[0]) < len(self.seqs[1]) else 1

		# Calculate length to normalize by
		if method == 'mismatch':
			length = sum(not self.gap_at(i) for i in range(len(self)))
		elif method == 'columns':
			length = len(self)
		elif method == 0:
			length = len(self.seqs[0])
		elif method == 1:
			length = len(self.seqs[1])
		else:
			raise ValueError('Invalid method {}'.format(repr(method)))

		# Return
		return float(identities) / length

	def iter_seq(self, seq):
		for i in range(len(self)):
			yield self.char_at(seq, i)

	def print_alignment(self, width=80, **kwargs):
		'''
		Pretty-prints the alignment.
		'''
		print_match = kwargs.pop('print_match', True)
		gapchar = kwargs.pop('gapchar', '-')

		stream = sys.stdout
		for p in range(0, len(self), width):
			lw = min(width, len(self) - p)

			self._print_seq(stream, 0, gapchar, (p, p + lw))
			stream.write('\n')

			if print_match:
				for i in range(p, p + lw):
					stream.write(self._match_char_at(i))
				stream.write('\n')

			self._print_seq(stream, 1, gapchar, (p, p + lw))
			stream.write('\n')

			stream.write('\n')

	def _print_seq(self, stream, seq, gapchar, idxs):
		for i in range(*idxs):
			if self.gaps[seq, i]:
				stream.write(gapchar),
			else:
				stream.write(self.char_at(seq, i))

	def _match_char_at(self, pos):
		if self.gaps[0, pos] or self.gaps[1, pos]:
			return ' '
		elif self.char_at(0, pos) == self.char_at(1, pos):
			return '|'
		else:
			return ':'
