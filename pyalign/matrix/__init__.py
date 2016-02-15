"""
Contains classes for representing substitution matrices, functions for
generating them or loading pre-calculated ones.
"""
from cPickle import Unpickler

from pkg_resources import resource_stream

from substitutionmatrix import *
from pam import pam_matrix


with resource_stream(__name__, 'matrices.pickle') as stream:
	unpickler = Unpickler(stream)
	matrices = unpickler.load()


def load_matrix(name):
	with resource_stream(__name__ + '.matrices', name + '.pickle') as stream:
		unpickler = Unpickler(stream)
		return unpickler.load()
