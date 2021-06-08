/*  TAlloc.h
 *  Header file with functions for data allocations and deallocations
 *  Jiri Filip, UTIA CAS CR, filipj"at"utia.cas.cz
 */

#ifndef TAlloc_c
#define TAlloc_c

 //allocations
float* Allocation1(int nrl, int nrh);
int* IAllocation1(int nrl, int nrh);
float** Allocation2(int nrl, int nrh, int ncl, int nch);
int** IAllocation2(int nrl, int nrh, int ncl, int nch);

//reallocations
float* Reallocation1(float* var, int nrl, int nrhOld, int nrhNew);
float** Reallocation2Rows(float** var, int nrl, int nrhOld, int nrhNew, int ncl, int nch);
int** IReallocation2Rows(int** var, int nrl, int nrhOld, int nrhNew, int ncl, int nch);

//deallocations
void Free1(float* m, int nrl, int nrh);
void IFree1(int* m, int nrl, int nrh);
void Free2(float** m, int nrl, int nrh, int ncl, int nch);
void IFree2(int** m, int nrl, int nrh, int ncl, int nch);

#endif // TAlloc_c

