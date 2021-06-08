/*!**************************************************************************
\file    AuxFuncs.h
\author  Jiri Filip
\date    15/11/2006
\version 1.01

  The header file for the:
	BTFBASE project with V. Havran (CVUT)

*****************************************************************************/
#ifndef AuxFuncs_c
#define AuxFuncs_c

//#################################################################################
//! \brief Global variable definitions
//#################################################################################

#define FLOATMAX ((float)(1e15))
#define MAXINT ((int)(32700))
#define EPSILON 3.0e-7
#define PI 3.14159265358979323846

#ifdef _MSC_VER
#define log2(A) log10((float)A)/log10(2.0f)
//#define isfinite(A) (!(_isnan(A)))
#endif

#define isfiniteBTF(A) (!(isnanf(A)))


// === float values

//! \brief reads file "fileName" and returns sizes [nr x nc] of stored 2D array
int ReadTxtHeader(const char* fileName, int* nr, int* nc, float* minV, float* maxV);

//! \brief reads 2D array of floats "arr" from file "fileName" and returns is sizes [nr x nc]
int ReadTxt(float** arr, const char* fileName, int* nr, int* nc);

// === integer values

//! \brief reads 2D array of ints "arr" from file "fileName" and returns is sizes [nr x nc]
int IReadTxtHeader(const char* fileName, int* nr, int* nc, int* minV, int* maxV);

//! \brief reads 2D array of ints "arr" from file "fileName" and returns is sizes [nr x nc]
int IReadTxt(int** arr, const char* fileName, int* nr, int* nc);

// Abort the compression for some reason, but before make some actions that
// could be useful, such as saving the databases + status
void AbortRun(int id);

#endif // AuxFuncs_c
