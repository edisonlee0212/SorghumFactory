/*!**************************************************************************
\file    TAlloc.cpp
\author  Jiri Filip
\date    April 2004
\version 1.00

  \brief Routines for data allocation and deallocation

<em>
COPYRIGHT:
Institute of Information Theory and Automation (UTIA),
Academy of Sciences of the Czech Republic, Prague 1997,
All rights reserved.

Please contact:
UTIA, Institute of Information Theory and Automation
Dr Michal Haindl
Pod vodarenskou vezi 4
18208 Praha 8
Czech Republic
</em>
******************************************************************************/


#include <cstdio>
#include <cstdlib> 
#include <cassert> 

//################################################################################
//# ALLOCATIONS                                                                 #
//################################################################################


float** Allocation2(int nrl, int nrh, int ncl, int nch)
{
	//! memory allocation, float matrix [nrl..nrh][ncl..nch 

	int j;
	float** m;
	if ((nrh - nrl + 1 <= 0) || (nch - ncl + 1 <= 0)) return NULL;

	m = (float**)calloc((unsigned)(nrh - nrl + 1), sizeof(float*));
	assert(m);
	if (!m) { printf(" allocation failure 1 in allocation2() \n"); }
	m -= nrl;

	for (j = nrl; j <= nrh; j++) {
		m[j] = (float*)calloc((unsigned)(nch - ncl + 1), sizeof(float));
		assert(m[j]);
		if (!m[j]) { printf(" allocation failure 2 in allocation2() \n"); }
		m[j] -= ncl;
	}

	return m;
} // allocation2 ----------------------------------------------- 

int** IAllocation2(int nrl, int nrh, int ncl, int nch)
{
	//! memory allocation, int matrix [nrl..nrh][ncl..nch] 
	int j;
	int** m;
	if ((nrh - nrl + 1 <= 0) || (nch - ncl + 1 <= 0)) return NULL;

	nch = nch;
	m = (int**)calloc((unsigned)(nrh - nrl + 1), sizeof(int*));
	if (!m) { printf(" allocation failure 1 in iallocation2\n"); }
	m -= nrl;

	for (j = nrl; j <= nrh; j++) {
		m[j] = (int*)calloc((unsigned)(nch - ncl + 1), sizeof(int));
		if (!m[j]) { printf(" allocation failure 2 in iallocation2\n"); }
		m[j] -= ncl;
	}
	return m;
} // iallocation2 ----------------------------------------------- 

//################################################################################
//# REALLOCATIONS                                                                 #
//################################################################################

float* Reallocation1(float* var, int nrl, int nrhOld, int nrhNew)
{
	//! memory reallocation, float matrix [nrl..nrh]
	assert(nrhOld < nrhNew);
	if ((nrhNew - nrl + 1 <= 0)) return NULL;
	float* m;
	m = (float*)calloc((unsigned)(nrhNew - nrl + 1), sizeof(float));
	if (!m) { printf(" allocation failure 1 in reallocation1() \n"); }
	// copy already used data
	for (int i = nrl; i <= nrhOld; i++)
		m[i] = var[i];

	free(var);
	return m;
} // reallocation1 ----------------------------------------------- 

float** Reallocation2Rows(float** var, int nrl, int nrhOld, int nrhNew, int ncl, int nch)
{
	//! memory reallocation, float matrix [nrl..nrh][ncl..nch]

	// we assume that we increase the number of rows only !!!
	assert(nrhOld < nrhNew);
	int j;
	float** m;
	if ((nrhNew - nrl + 1 <= 0) || (nch - ncl + 1 <= 0)) return NULL;

	m = (float**)calloc((unsigned)(nrhNew - nrl + 1), sizeof(float*));
	if (!m) { printf(" allocation failure 1 in reallocation2rows() \n"); }
	m -= nrl;
	// copy the pointers to already used data
	for (j = nrl; j <= nrhOld; j++) {
		m[j] = var[j];
	}

	for (j = nrhOld + 1; j <= nrhNew; j++) {
		m[j] = (float*)calloc((unsigned)(nch - ncl + 1), sizeof(float));
		if (!m[j]) { printf(" allocation failure 2 in reallocation2rows() \n"); }
		m[j] -= ncl;
	}
	free((float**)(var + nrl));
	return m;
} // reallocation2rows ----------------------------------------------- 

int** IReallocation2Rows(int** var, int nrl, int nrhOld, int nrhNew, int ncl, int nch)
{
	//! memory reallocation, int matrix [nrl..nrh][ncl..nch]

	// we assume that we increase the number of rows only !!!
	assert(nrhOld < nrhNew);
	int j;
	int** m;
	if ((nrhNew - nrl + 1 <= 0) || (nch - ncl + 1 <= 0)) return NULL;

	m = (int**)calloc((unsigned)(nrhNew - nrl + 1), sizeof(int*));
	if (!m) { printf(" allocation failure 1 in reallocation2rows() \n"); }
	m -= nrl;
	// copy the pointers to already used data
	for (j = nrl; j <= nrhOld; j++) {
		m[j] = var[j];
	}

	for (j = nrhOld + 1; j <= nrhNew; j++) {
		m[j] = (int*)calloc((unsigned)(nch - ncl + 1), sizeof(int));
		if (!m[j]) { printf(" allocation failure 2 in reiallocation2rows() \n"); }
		m[j] -= ncl;
	}
	free((int**)(var + nrl));
	return m;
} // reiallocation2rows ----------------------------------------------- 

//################################################################################
//# DEALLOCATIONS                                                                 #
//################################################################################

void Free2(float** m, int nrl, int nrh, int ncl, int nch)
{
	//! frees memory allocated by allocation2 
	int i;
	//  nch=nch;
	//  nrh=nrh;

	for (i = nrh; i >= nrl; i--) free((float*)(m[i] + ncl));
	free((float**)(m + nrl));
} // freemem2 --------------------------------------------------- 

void IFree2(int** m, int nrl, int nrh, int ncl, int nch)
{
	//! This method frees memory iallocated by iallocation2 
	int i;

	nch = nch;
	for (i = nrh; i >= nrl; i--) free((int*)(m[i] + ncl));
	free((int**)(m + nrl));
} // freeimem2 ---------------------------------------------------

