/*
 * Header file for pwglobalalign.c
 * - Imports standard, Python, and Numpy libraries
 * - #defines a few constants
 * - Defines structs and declares related functions
 * - Declares main alignment functions
 */

// Standard libraries
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <string.h>

// Define this to avoid warnings about deprecated numpy API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Main python header and numpy type header
#include <Python.h>
#include <numpy/ndarraytypes.h>



/*
 * Constants defining edit actions used to transform one sequence into another.
 * An array of these can be used to define an alignment between two sequences.
 * These also correspond to directions in a Needleman-Wunch path matrix. Note
 * that values are chosen so that they can be combined via bitwise binary
 * oparations.
 * 
 * The proper way to do this might be to do with with typedef enum {...}
 * instead of #defines but it seems enums in C take up as much space as an
 * int... this is problematic as this matrix is the biggest memory allocation
 * for the algorithm so using bytes instead of 32-bit ints cuts memory usage
 * by 4x.
 */
#define EDIT_SUB 1 // 2^0 - No gap - Diagonal
#define EDIT_INS 2 // 2^1 - Gap in seq 1 - Right
#define EDIT_DEL 4 // 2^2 - Gap in seq 2 - Down
typedef uint8_t seq_edit;



/**
 * Stores information on a scoring method for pairwise alignment using a
 * substitution matrix and affine gap penalties. Note that the matrix should
 * be a similarity matrix, meaning better matches have larger positive entries.
 * A gap of length n will be penalized by [o + e*(k-1)], where o and e are
 * both nonnegative real numbers.
 * @prop {*npy_float} matrix Pointer to symmetric 256x256 matrix of numpy
 *                           floats. Entry [i][j] is score for matching symbol
 *                           i with symbol j.
 * @prop {npy_float} o Gap open penalty (non-negative).
 * @prop {npy_float} e Gap extension penalty (non-negative).
 */
struct PairwiseScoringMethod {
	npy_float *matrix;
	npy_float o;
	npy_float e;
};


/**
 * Globally aligns two sequnces via Gotoh's algorithm (variant on NW/SM that
 * allows for affine gap penalties). Yields a traceback array of sequence
 * edit actions. This is equivalent to the *reversed* alignment.
 * @param {npy_ubyte*} seqA Pointer to array containing sequence A.
 * @param {npy_intp} lenA Length of sequence A.
 * @param {npy_ubyte*} seqA Pointer to array containing sequence B.
 * @param {npy_intp} lenA Length of sequence B.
 * @param {struct PairwiseScoringMethod*} sm Scoring method.
 * @param {seq_edit*} tbOut Pointer to array of sequence edits of length
 *                             (lenA + lenB) that the traceback will be
 *                             written to.
 * @returns {npy_inpt} Length of traceback, or -1 on error.
 */
npy_intp gotohAlign(npy_ubyte *seqA, npy_intp lenA,
                    npy_ubyte *seqB, npy_intp lenB,
                    struct PairwiseScoringMethod *sm, seq_edit *tbOut);


/**
 * Does most of the work of the Gotoh algorithm. Fills in matrix with
 * allowable edit actions for each cell (bitwise combinations of seq_edit
 * values).
 * @param {npy_ubyte*} seqA Pointer to array containing sequence A.
 * @param {npy_intp} lenA Length of sequence A.
 * @param {npy_ubyte*} seqA Pointer to array containing sequence B.
 * @param {npy_intp} lenA Length of sequence B.
 * @param {struct PairwiseScoringMethod*} sm Scoring method.
 * @param {seq_edit*} editMatrix Pointer to multidimensional array of
 *                               seq_edit of size lenA * lenB that will be
 *                               filled in with edit values. Should be
 *                               initially all 0.
 */
void fillEditMatrix(npy_ubyte *seqA, npy_intp lenA,
                    npy_ubyte *seqB, npy_intp lenB,
                    struct PairwiseScoringMethod *sm,
                    seq_edit *editMatrix);


/**
 * Backtracks through the edit matrix, filling a traceback array with sequence
 * edit actions (starting from the end of the alignment and working
 * backwards).
 * @param {npy_ubyte*} seqA Pointer to array containing sequence A.
 * @param {npy_intp} lenA Length of sequence A.
 * @param {npy_ubyte*} seqA Pointer to array containing sequence B.
 * @param {npy_intp} lenA Length of sequence B.
 * @param {struct PairwiseScoringMethod*} sm Scoring method.
 * @param {seq_edit*} editMatrix Pointer to multidimensional array of
 *                               seq_edit of size lenA * lenB containing the
 *                               edit values.
 * @param {seq_edit*} tbOut Pointer to array of seq_edit of length
 *                          (lenA + lenB) that the traceback will be
 *                          written to.
 * @returns {npy_intp} Length of traceback, or -1 on error.
 */
npy_intp makeTraceback(npy_ubyte *seqA, npy_intp lenA,
                       npy_ubyte *seqB, npy_intp lenB,
                       struct PairwiseScoringMethod *sm,
                       seq_edit *editMatrix, seq_edit *tbOut);


/**
 * Converts a traceback array to a 2d matrix of gaps representing the
 * alignment. The traceback array gives edits in reverse order, but the
 * gaps will start from the beginning.
 * @param {seq_edit*} tb Pointer to beginning of traceback, array of seq_edit
 *                       values.
 * @param {npy_intp} tbLen Length of traceback.
 * @param {npy_bool*} gapsOut Pointer to 2d array to write gaps to. Must be
 *                            of shape (2, tbLen) (row-major order).
 */
void tracebackToGaps(seq_edit *tb, npy_intp tbLen, npy_bool *gapsOut);
