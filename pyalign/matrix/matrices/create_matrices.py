"""Script that formats substitution matrices obtained from NCBI

This folder contains substitution matrices (mostly PAM and BLOSUM) used by
NCBI's BLAST tool. Data was obtained from
ftp://ftp.ncbi.nih.gov/blast/matrices/ on 2/14/16. 


* PAM matrices - in two formats: PAM{n} and PAM{n}.cdi. N ranges from 10 to
	500 in increments of 10 for the first. Scales range from ln(2)/2 for
	PAM10 to ln(2)/7 for PAM500. The .cdi variants only exist for a few N
	values and seem to be in units of 10ths of a bit.
* BLOSUM matrices - from BLOSUM30 to 100 in increments of 5. There's also a
	BLOSUMN matrix which I am unsure about. Each has a .50 variant which is
	the exact same file (by sha1 checksum).
* Nucleotide - NUC.4.4 looks useful, NUC.4.2 is non-rectangular?
* Identity - IDENTITY and MATCH can be replaced by
	pyalign.matrix.identity_matrix.
* Others - DAYHOFF and GONNET seem to be slightly different PAM250 matrices.
	Don't see these being useful.

"""

import os
from cStringIO import StringIO
from cPickle import Pickler, HIGHEST_PROTOCOL

import numpy as np
import pandas as pd


from pyalign import SubstitutionMatrix


ln2 = float(np.log(2))


# Names of all PAM matrix files along with their mutation distances and scales
# in inverse bits
pam_files = [
	('PAM10', 10, 2),
	('PAM20', 20, 2),
	('PAM30', 30, 2),
	('PAM40', 40, 2),
	('PAM50', 50, 2),
	('PAM60', 60, 2),
	('PAM70', 70, 2),
	('PAM80', 80, 2),
	('PAM90', 90, 2),
	('PAM100', 100, 2),
	('PAM110', 110, 2),
	('PAM120', 120, 2),
	('PAM130', 130, 2),
	('PAM140', 140, 2),
	('PAM150', 150, 2),
	('PAM160', 160, 2),
	('PAM170', 170, 3),
	('PAM180', 180, 3),
	('PAM190', 190, 3),
	('PAM200', 200, 3),
	('PAM210', 210, 3),
	('PAM220', 220, 3),
	('PAM230', 230, 3),
	('PAM240', 240, 3),
	('PAM250', 250, 3),
	('PAM260', 260, 3),
	('PAM270', 270, 4),
	('PAM280', 280, 4),
	('PAM290', 290, 4),
	('PAM300', 300, 4),
	('PAM310', 310, 4),
	('PAM320', 320, 4),
	('PAM330', 330, 4),
	('PAM340', 340, 4),
	('PAM350', 350, 5),
	('PAM360', 360, 5),
	('PAM370', 370, 5),
	('PAM380', 380, 5),
	('PAM390', 390, 5),
	('PAM400', 400, 5),
	('PAM410', 410, 6),
	('PAM420', 420, 6),
	('PAM430', 430, 6),
	('PAM440', 440, 6),
	('PAM450', 450, 6),
	('PAM460', 460, 6),
	('PAM470', 470, 7),
	('PAM480', 480, 7),
	('PAM490', 490, 7),
	('PAM500', 500, 7),
	('PAM40.cdi', 40, 10),
	('PAM80.cdi', 80, 10),
	('PAM120.cdi', 120, 10),
	('PAM160.cdi', 160, 10),
	('PAM200.cdi', 200, 10),
	('PAM250.cdi', 250, 10)
]

# Names of all PAM matrix files along with their percent values and scales in
# inverse bits
blosum_files = [
	('BLOSUM30', 30, 5),
	('BLOSUM35', 35, 4),
	('BLOSUM40', 40, 4),
	('BLOSUM45', 45, 3),
	('BLOSUM50', 50, 3),
	('BLOSUM55', 55, 3),
	('BLOSUM60', 60, 2),
	('BLOSUM62', 62, 2),
	('BLOSUM65', 65, 2),
	('BLOSUM70', 70, 2),
	('BLOSUM75', 75, 2),
	('BLOSUM80', 80, 3),
	('BLOSUM85', 85, 2),
	('BLOSUM90', 90, 2),
	('BLOSUM100', 100, 3),
	('BLOSUMN', None, 2)
 ]

# Names of additional matrix files in format (file_name, new_name, attrs)
addl_files = [
	('NUC', 'NUC.4.4', {'type': 'NUC'})
]



def parse_ncbi_matrix(lines):
	"""Parses a matrix file in the format obtained from NCBI

	returns a tuple of (matrix, symbols, description)
	"""

	# Buffer to store uncommented lines
	table_buffer = StringIO()

	# Store description as well
	desc_lines = []

	# Split lines into description/non-description
	for line in lines:
		if line.startswith('#'):
			desc_line = line[1:]
			if desc_line.startswith(' '):
				desc_line = desc_line[1:]
			if desc_line:
				desc_lines.append(desc_line)

		elif line.strip():
			table_buffer.write(line)

	# Parse table
	table_buffer.seek(0)
	table = pd.read_table(table_buffer, sep=r'\s+')

	# Should have identical row/column labels
	assert table.columns.equals(table.index)

	return table.values, list(table.columns), ''.join(desc_lines).strip()


def get_matrices():
	"""(name, ncbi_file, extra_attrs) for each matrix to be formatted"""

	matrices = list(addl_files)

	# PAM matrices
	for fname, dist, bits in pam_files:
		name = 'PAM{}_{}'.format(dist, bits)
		attrs = {
			'type': 'PAM',
			'scale': ln2 / bits,
			'scale_invbits': bits,
			'dist': dist
		}
		matrices.append((name, fname, attrs))

	# BLOSUM matrices
	for fname, pct, bits in blosum_files:
		attrs = {
			'type': 'BLOSUM',
			'scale': ln2 / bits,
			'scale_invbits': bits,
			'percent': pct
		}
		matrices.append((fname, fname, attrs))

	return matrices


def create_matrices(indir='.'):
	"""Creates SubstitutionMatrix instances from NCBI matrices in directory.

	Also gives dict of matrix attributes by name.

	Returns:
		dict (matrices), dict (attributes)
	"""

	matrices = dict()

	# Stores attributes for each matrix
	matrix_attrs = dict()

	# For each matrix
	for name, ncbi_file, extra_attrs in get_matrices():

		# Parse the file
		fpath = os.path.join(indir, ncbi_file)
		with open(fpath) as fh:
			values, symbols, description = parse_ncbi_matrix(fh)

		# Create the matrix object
		matrix = SubstitutionMatrix(symbols, values)
		matrices[name] = matrix

		# Attributes
		attrs = {
			'ncbi_file': ncbi_file,
			'description': description,
			'range': (np.min(values), np.max(values))
		}
		attrs.update(extra_attrs)
		matrix_attrs[name] = attrs

	return matrices, matrix_attrs


def pickle_matrices(matrices, outdir='.'):
	"""Pickles dictionary of matrices output by create_matrices"""
	for name, matrix in matrices.iteritems():
		fpath = os.path.join(outdir, name + '.pickle')
		with open(fpath, 'wb') as fh:
			pickler = Pickler(fh, HIGHEST_PROTOCOL)
			pickler.dump(matrix)
