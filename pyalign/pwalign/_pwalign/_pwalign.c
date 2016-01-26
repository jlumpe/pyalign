#include "pwalign.h"

#include <numpy/arrayobject.h>


// Docstrings
static char module_docstring[] = "C module for pairwise alignment";
static char pwalign_docstring[] = "\
Performs global optimal pairwise alignment of two sequences using Gotoh's \
algorithm";


// pwalign python method declaration
static PyObject *pwalign_pwalign(PyObject *self, PyObject *args);


// Helper function declarations
static bool pwalignGetArgArrays(PyObject *seqAObj, PyObject *seqBObj,
                                PyObject *matrixObj, PyArrayObject **seqANpPtr,
                                PyArrayObject **seqBNpPtr,
                                PyArrayObject **matrixNpPtr);


// Module methods
static PyMethodDef module_methods[] = {
	{"pwalign", pwalign_pwalign, METH_VARARGS, pwalign_docstring},
	{NULL, NULL, 0, NULL}
};


// Module initialization
PyMODINIT_FUNC init_pwalign(void)
{
	PyObject *m = Py_InitModule3("_pwalign", module_methods, module_docstring);
	if(m == NULL)
		return;

	// Load numpy functionality
	import_array();
}


// pwalign python method definition
static PyObject *pwalign_pwalign(PyObject *self, PyObject *args)
{
	PyObject *seqAObj, *seqBObj, *matrixObj;
	PyArrayObject *seqANp, *seqBNp, *matrixNp;
	float gap_open, gap_extend;
	npy_ubyte *seqA, *seqB;
	npy_intp lenA, lenB;
	struct PairwiseScoringMethod sm;
	npy_intp tbLen;
	npy_float alignScore;
	PyArrayObject *gapArray;

	// Parse input tuple
	if(!PyArg_ParseTuple(args, "OOOff", &seqAObj, &seqBObj, &matrixObj, &gap_open, &gap_extend))
		return NULL;

	// Interpret input objects as Numpy arrays
	if(!pwalignGetArgArrays(seqAObj, seqBObj, matrixObj, &seqANp, &seqBNp, &matrixNp))
		return NULL;

	// Get sequence arrays and length
	seqA = (npy_ubyte*)PyArray_DATA(seqANp);
	seqB = (npy_ubyte*)PyArray_DATA(seqBNp);
	lenA = PyArray_DIM(seqANp, 0);
	lenB = PyArray_DIM(seqBNp, 0);

	// Create scoring method
	sm.matrix = (npy_float*)PyArray_DATA(matrixNp);
	sm.o = gap_open;
	sm.e = gap_extend;

	// Run the alignment algorithm and get a traceback array
	seq_edit traceback[lenA + lenB];
	tbLen = gotohAlign(seqA, lenA, seqB, lenB, &sm, traceback, &alignScore);

	// Check if alignment successful, if so create gap array
	if(tbLen != -1)
	{
		npy_intp gapDims[2] = {2, tbLen};
		gapArray = (PyArrayObject*)PyArray_SimpleNew(2, gapDims, NPY_UBYTE);
	}
	else
	{
		gapArray = NULL;
	}

	// Only proceed to fill it if there haven't been any erros so far
	// (not actually sure, but I think PyArray_SimpleNew() can return NULL
	// on failure, so try to catch that)
	if(gapArray != NULL)
	{
		// Get pointer to array data and increment reference count
		npy_bool *gapData = (npy_bool*)PyArray_DATA(gapArray);
		Py_INCREF(gapArray);

		// Fill in gap array
		tracebackToGaps(traceback, tbLen, gapData);
	}

	// Decrement reference counts on Python objects that were created in this
	// function (regardless of whether creating the gap array succeeded or not)
	Py_DECREF(seqANp);
	Py_DECREF(seqBNp);
	Py_DECREF(matrixNp);

	// Build and return output tuple
	if(gapArray != NULL)
	{
		float alignScoreFloat = (float)alignScore;
		PyObject *ret = Py_BuildValue("Of", gapArray, alignScoreFloat);
		return ret;
	}
	else
	{
		return NULL;
	}
}

static bool pwalignGetArgArrays(PyObject *seqAObj, PyObject *seqBObj,
                                PyObject *matrixObj, PyArrayObject **seqANpPtr,
                                PyArrayObject **seqBNpPtr,
                                PyArrayObject **matrixNpPtr)
{
	bool arrayError;

	// Get as array objects
	PyArrayObject *seqANp = (PyArrayObject*)PyArray_FROM_OTF(seqAObj,
		NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *seqBNp = (PyArrayObject*)PyArray_FROM_OTF(seqBObj,
		NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
	PyArrayObject *matrixNp = (PyArrayObject*)PyArray_FROM_OTF(matrixObj,
		NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

	// Check error conditions
	arrayError = false;
	if(seqANp == NULL || seqBNp == NULL || matrixNp == NULL)
	{
		arrayError = true;
	}
	else if(PyArray_NDIM(seqANp) != 1)
	{
		PyErr_SetString(PyExc_ValueError, "Bad number of dimensions on sequnce 1");
		arrayError = true;
	}
	else if(PyArray_NDIM(seqBNp) != 1)
	{
		PyErr_SetString(PyExc_ValueError, "Bad number of dimensions on sequnce 2");
		arrayError = true;
	}
	else if(PyArray_NDIM(matrixNp) != 2)
	{
		PyErr_SetString(PyExc_ValueError, "Bad number of dimensions on matrix");
		arrayError = true;
	}
	else
	{
		npy_intp *matrixShape = PyArray_DIMS(matrixNp);
		if(matrixShape[0] != 256 || matrixShape[1] != 256)
		{
			PyErr_SetString(PyExc_ValueError, "Bad shape on matrix - "
											  "should be (256, 256)");
			arrayError = true;
		}
	}
	
	// Clean up and return if error
	if(arrayError)
	{
		Py_XDECREF(seqANp);
		Py_XDECREF(seqBNp);
		Py_XDECREF(matrixNp);
		return false;
	}
	else
	{
		*seqANpPtr = seqANp;
		*seqBNpPtr = seqBNp;
		*matrixNpPtr = matrixNp;
		return true;
	}
}
