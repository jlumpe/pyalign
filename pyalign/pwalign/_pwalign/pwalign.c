#include "pwalign.h"


npy_float max(npy_float val1, npy_float val2)
{
	if(val1 >= val2)
		return val1;
	else
		return val2;
}


npy_intp gotohAlign(npy_ubyte *seqA, npy_intp lenA,
                    npy_ubyte *seqB, npy_intp lenB,
                    struct PairwiseScoringMethod *sm, seq_edit *tbOut)
{
	seq_edit *editMatrix;
	size_t matSize;

	// Fill in edit matrix
	matSize = sizeof(seq_edit) * lenA * lenB;
	editMatrix = malloc(matSize);
	if(editMatrix == NULL)
		return 0;

	memset(editMatrix, 0, matSize);

	fillEditMatrix(seqA, lenA, seqB, lenB, sm, editMatrix);

	// Now do the backtracking
	return makeTraceback(seqA, lenA, seqB, lenB, sm, editMatrix, tbOut);
}


void fillEditMatrix(npy_ubyte *seqA, npy_intp lenA,
                    npy_ubyte *seqB, npy_intp lenB,
                    struct PairwiseScoringMethod *sm,
                    seq_edit *editMatrix)
{
	npy_intp m, n, i;
	npy_ubyte charA, charB;
	npy_float subScore, sCurrent, sDiag, sLeft, sAbove;
	npy_float pCurrent, pAbove;
	npy_float qCurrent, qLeft;
	npy_float sRow[lenB], pRow[lenB];

	// Initialize row arrays (values of last row = top row of matrices)
	sRow[0] = -sm->o;
	pRow[0] = -INFINITY;
	for(i = 1; i < lenB; i++)
	{
		sRow[i] = sRow[i - 1] - sm->e;
		pRow[i] = -INFINITY;
	}

	// Loop over A
	for(m = 0; m < lenA; m++)
	{
		if(m == 0)
		{
			sDiag = 0; // S_{m-1,n-1}
			sLeft = -sm->o; // S_{m, n-1}
		}
		else
		{
			sDiag = -sm->o - m * sm->e;
			sLeft = sDiag - sm->e;
		}

		// Q_{m,n-1}
		qLeft = -INFINITY;

		// A_m
		charA = seqA[m];

		// Loop over B
		for(n = 0; n < lenB; n++)
		{
			charB = seqB[n];

			sAbove = sRow[n];
			pAbove = pRow[n];

			// Score of a substitution/match
			subScore = sDiag + sm->matrix[(256 * charA) + charB];

			// Score of an insertion - current value of P matrix
			pCurrent = max(sAbove - sm->o, pAbove - sm->e);

			// Score of a deletion - current value of Q matrix
			qCurrent = max(sLeft - sm->o, qLeft - sm->e);

			// Max of all 3 - current value of S matrix
			sCurrent = max(subScore, max(pCurrent, qCurrent));

			// Check which ones match the max - fill in direction matrix
			npy_float tol = 1e-3;
			if(subScore >= sCurrent - tol)
				editMatrix[m * lenB + n] |= EDIT_SUB;
			if(pCurrent >= sCurrent - tol)
				editMatrix[m * lenB + n] |= EDIT_DEL;
			if(qCurrent >= sCurrent - tol)
				editMatrix[m * lenB + n] |= EDIT_INS;

			// Update left and diagonal values for next column (m -> m+1)
			sLeft = sCurrent;
			qLeft = qCurrent;
			sDiag = sAbove;

			// Update current row variables
			sRow[n] = sCurrent;
			pRow[n] = pCurrent;
		}
	}
}


npy_intp makeTraceback(npy_ubyte *seqA, npy_intp lenA,
                       npy_ubyte *seqB, npy_intp lenB,
                       struct PairwiseScoringMethod *sm,
                       seq_edit *editMatrix, seq_edit *tbOut)
{
	npy_intp m, n, i;
	seq_edit current, last, cellVal;

	// Start at lower-right of matrix
	m = lenA - 1;
	n = lenB - 1;

	// Backtrack until we hit an edge
	i = 0;
	last = 0;
	while(m >= 0 && n >= 0)
	{
		// Check if we could do a SUB, INS, or EDIT action,
		// preferentially if is the same as the last action
		cellVal = editMatrix[m * lenB + n];
		if(cellVal & last & EDIT_SUB)
			current = EDIT_SUB;
		else if(cellVal & last & EDIT_INS)
			current = EDIT_INS;
		else if(cellVal & last & EDIT_DEL)
			current = EDIT_DEL;
		else if(cellVal & EDIT_SUB)
			current = EDIT_SUB;
		else if(cellVal & EDIT_INS)
			current = EDIT_INS;
		else if(cellVal & EDIT_DEL)
			current = EDIT_DEL;
		else
		{
			PyErr_Format(
				PyExc_RuntimeError,
				"Invalid value %i at position (%i, %i) in edit matrix.",
				(int)cellVal, (int)m, (int)n
			);
			return -1;
		}

		// Fill in traceback value
		tbOut[i++] = current;
		last = current;

		// Move to next position in matrix
		if(current & (EDIT_SUB | EDIT_DEL))
			m--;
		if(current & (EDIT_SUB | EDIT_INS))
			n--;
	}

	// Backtrack from edge to corner
	while(m-- >= 0)
		tbOut[i++] = EDIT_DEL;
	while(n-- >= 0)
		tbOut[i++] = EDIT_INS;

	// Return length of traceback
	return i;
}

void tracebackToGaps(seq_edit *tb, npy_intp tbLen, npy_bool *gapsOut)
{
	for(int i = 0; i < tbLen; i++)
	{
		int j = tbLen - i - 1;
		if(tb[i] == EDIT_INS)
			gapsOut[j] = NPY_TRUE;
		else
			gapsOut[j] = NPY_FALSE;
		if(tb[i] == EDIT_DEL)
			gapsOut[j + tbLen] = NPY_TRUE;
		else
			gapsOut[j + tbLen] = NPY_FALSE;
	}
}
