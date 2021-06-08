/*!**************************************************************************
\file    PDF1D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  1D PDF database
******************************************************************************/

#ifndef PDF1D_c
#define PDF1D_c

#include <GlobalDefs.hpp>
#include <SharCoors.hpp>

//############################################################################
//! \brief PDF1 - database of Luminance slices
//############################################################################

class CPDF1D {
private:
	// the number of allocated 1D functions
	int m_maxPdf1D;
	// the number of values for 1D function
	int m_lengthOfSlice;
	// the data array of 1D functions. These are normalized !
	float** m_pdf1DBasis;
	// current number of stored 1D functions
	int m_numOfPdf1D;
	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database

public:
	// constructor, allocates maximum number of slices, the size of 1D function,
	// the metric used for comparison, and maximum shift for compression
	CPDF1D(int maxPDF1D, int LengthOfSlice, int metric, int maxShift);
	~CPDF1D();

	// delete the data from the database, but not from the databases below
	void DeleteData();

	// check & reallocation of database if needed
	void Reallocate();

	// computes the sum of the array
	float NormalPdf1D(const float* const lums);

	// get a single value at the discrete position
	inline float Get(int sliceIndex, int posBeta, TSharedCoordinates& tc) const;
	inline void GetAll(int sliceIndex, TSharedCoordinates& tc) const;

	// get single value for arbitrary angle beta and slice index
	float GetVal(int sliceIndex, float beta, TSharedCoordinates& tc) const;
	// Here beta is specified by 'tc'
	float GetVal(int sliceIndex, TSharedCoordinates& tc) const;
	void GetValShepard(int sliceIndex, float scale, float sumDist2,
		float userCmData[], TSharedCoordinates& tc) const;

	// returns No. of stored 1D functions
	int GetNumOfPdf1D() const;// S_1
	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;

	// returns the length of slice
	int GetSliceLength() const;

	int Load(char* prefix, int mlf, int algPut);

};//--- CPDF1D ---------------------------------------------------------

inline float
CPDF1D::Get(const int sliceIndex, const int posBeta, TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_numOfPdf1D));
	assert((posBeta >= 0) && (posBeta < m_lengthOfSlice));
	return m_pdf1DBasis[sliceIndex][posBeta];
}//--- get -----------------------------------------------------------

inline void
CPDF1D::GetAll(const int sliceIndex, TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_numOfPdf1D));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_lengthOfSlice));

	tc.m_scale *= m_pdf1DBasis[sliceIndex][tc.m_indexBeta];
}//--- get -----------------------------------------------------------

#endif
