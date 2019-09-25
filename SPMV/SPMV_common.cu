#include <assert.h>     /* assert */

void fill(float *A, const int n, const float maxi)
{
    for (int j = 0; j < n; j++)
    {
        A[j] = ((float) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
} 

void initRandomMatrix_ver2(int *cols, int *rowDelimiters, const int n, const int dim)
{
	int row, prev_row, ind, cont, elem;
	
	int stride = dim * dim / n;  
	prev_row=-1;
	cont = 0;
	for (elem = 0; elem<n ; elem++) {
		ind = elem * stride;
		row = ind / dim;
		if (row != prev_row){
			rowDelimiters[cont] = elem;
			prev_row = row;
			cont++;
		}
		cols[elem]= ind % dim;
	}
	 
	rowDelimiters[cont] = elem-1;
		
}

void initRandomMatrix_ver3(int *cols, int *rowDelimiters, const int n, const int dim)
{
	int row, prev_row, ind, cont, elem, r;
	int elem_per_row = n / dim;
	
	rowDelimiters[0]= 0;
	cont = 0;
	for (r =0 ; r<dim; r++){
		int ini = r - elem_per_row;
		int end = r + elem_per_row;
		
		if (ini < 0 )
			ini = 0;
		
		if (end > dim)
			ini = dim - elem_per_row-1;
		
		for (int elem=0;elem<elem_per_row; elem++)
				cols[cont++] = ini++;
				
		rowDelimiters[r+1] = cont;
		
	}
	
	return;
}

void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
	long long ldim, ln, i, j ;
    long long  nnzAssigned = 0;

	ln = n;
	ldim = dim;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)ln / ((double)ldim * (double)ldim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (i = 0; i < ldim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (j = 0; j < ldim; j++)
        {
            long long numEntriesLeft = (ldim * ldim) - ((i * ldim) + j);
            long long needToAssign   = ln - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < ln && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Runs sparse matrix vector multiplication on the CPU
//
// Arguements:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element of A
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last
//                  element of A
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: input - buffer of size dim
//        output - result from the spmv calculation
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing directly
//   out indirectly through a pointer
// ****************************************************************************

void spmvCpu(const float *val, const int *cols, const int *rowDelimiters,
	     const float *vec, int dim, float *out)
{
    for (int i=0; i<dim; i++)
    {
        float t = 0;
        for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
        {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}


// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of GPU results by comparing to CPU results
//
// Arguments:
//   cpuResults: array holding the CPU result vector
//   gpuResults: array hodling the GPU result vector
//   size: number of elements per vector
//   pass: optional iteration number
//
// Programmer: Lukasz Wesolowski
// Creation: June 23, 2010
// Returns:
//   nothing
//   prints "Passed" if the vectors agree within a relative error of
//   MAX_RELATIVE_ERROR and "FAILED" if they are different
// ****************************************************************************

bool verifyResults(const float *cpuResults, const float *gpuResults,
                   const int size)
{
    bool passed = true;
    for (int i = 0; i < size; i++)
    {
        if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i]
            > 0.02)
        {
//            cout << "Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
//                " dev: " << gpuResults[i] << endl;
            passed = false;
        }
    }
 
    return passed;
}