#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <AuxFuncs.hpp>

int ReadTxtHeader(const char* fileName, int* nr, int* nc, float* minV, float* maxV)
/* reads file "fileName" and returns sizes [nr x nc] of stored 2D array*/
{
	FILE* fp;
	if ((fp = fopen(fileName, "r")) == NULL) {
		printf("Error - opening file %s !!!\n", fileName);
		return -1;
	}

	int v = fscanf(fp, "%d %d %f %f\n", nr, nc, minV, maxV); v = v;
	assert(v == 4);
	fclose(fp);

	return 0; // ok
}//--- readTXTheader --------------------------------------------------

int ReadTxt(float** arr, const char* fileName, int* nr, int* nc)
/* reads 2D array of floats "arr" from file "fileName" and returns is sizes [nr x nc]*/
{
	FILE* fp;
	if ((fp = fopen(fileName, "r")) == NULL) {
		printf("Error - opening file %s !!!\n", fileName);
		return -1;
	}

	float tmp1, tmp2;
	int v = fscanf(fp, "%d %d %f %f\n", nr, nc, &tmp1, &tmp2); v = v;
	assert(v == 4);
	for (int irow = 0; irow < *nr; irow++) {
		for (int jcol = 0; jcol < *nc; jcol++) {
			v = fscanf(fp, "%f ", &arr[irow][jcol]);
			assert(v == 1);
		}
		v = fscanf(fp, "\n");
	}
	fclose(fp);
	return 1;
}//--- readTXT ------------------------------------------------------

int IReadTxtHeader(const char* fileName, int* nr, int* nc, int* minV, int* maxV)
/* reads file "fileName" and returns sizes [nr x nc] of stored 2D array*/
{
	FILE* fp;
	if ((fp = fopen(fileName, "r")) == NULL) {
		printf("Error - opening file %s !!!\n", fileName);
		return -1;
	}

	int v = fscanf(fp, "%d %d %d %d\n", nr, nc, minV, maxV); v = v;
	assert(v == 4);
	fclose(fp);

	return 0; // ok
}//--- readTXTheader --------------------------------------------------


int IReadTxt(int** arr, const char* fileName, int* nr, int* nc)
//! \brief reads 2D array of ints "arr" from file "fileName" and returns is sizes [nr x nc]
{
	FILE* fp;
	if ((fp = fopen(fileName, "r")) == NULL) {
		printf("Error - opening file %s !!!\n", fileName);
		return -1;
	}

	int tmp1, tmp2;
	int v = fscanf(fp, "%d %d %d %d\n", nr, nc, &tmp1, &tmp2); v = v;
	assert(v == 4);
	for (int irow = 0; irow < *nr; irow++) {
		for (int jcol = 0; jcol < *nc; jcol++) {
			v = fscanf(fp, "%d ", &arr[irow][jcol]);
			assert(v == 1);
		}
		v = fscanf(fp, "\n");
	}
	fclose(fp);
	return 1;
}//--- ireadTXT ------------------------------------------------------

// Abort the compression for some reason, but before make some actions that
// could be useful, such as saving the databases + status
void
AbortRun(int id)
{
	printf("Aborting the program, abort id = %d\n", id);
	abort();
}

