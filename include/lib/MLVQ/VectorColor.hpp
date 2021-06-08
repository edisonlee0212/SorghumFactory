/*!**************************************************************************
\file    VectorColor.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  CIE ab colours database
******************************************************************************/

#ifndef VECTORCOLOR_c
#define VECTORCOLOR_c

#include <SharCoors.hpp>

//#################################################################################
//! \brief VectorColor - database of user color model
//#################################################################################

class CVectorColor
{
private:
	// the index from which we start the search 
	int m_startIndex;
	// no. of channels describing one color (in our case usually 2 (CIE a-b))
	int m_numOfChannels;
	// the number of allocated a-b colors
	int m_maxVectorColor;
	int m_initialMaxVectorColor;
	// the data array of a-b colors
	float** m_vectorColorBasis;
	// current number of stored a-b colors
	int m_noOfColors;
	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database
public:
	// constructor, allocates maximum number of colors
	CVectorColor(int maxVectorColor);
	~CVectorColor();

	void DeleteData();

	// check & reallocation of database if needed
	void Reallocate();

	// get a single colour value specified by index and posAB (0,1)
	inline float Get(int colorIndex, int posAB, TSharedCoordinates& tc) const;
	inline void GetAll(int colorIndex, float LAB[], TSharedCoordinates& tc) const;
	inline void GetAll(int colorIndex, TSharedCoordinates& tc) const;
	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;
	// returns No. of stored colours
	int GetNoOfColors() const; //S _2 .. the number of stored colours in the database

	int Load(char* prefix, int MLF, int algPut);
};//--- CVectorColor ---------------------------------------------------------

inline float
CVectorColor::Get(int colorIndex, int posAB, TSharedCoordinates& tc) const
{
	assert((posAB == 0) || (posAB == 1));
	assert((colorIndex >= 0) && (colorIndex < m_maxVectorColor));

	return m_vectorColorBasis[colorIndex][posAB];
}//--- get -----------------------------------------------------------

inline void
CVectorColor::GetAll(int colorIndex, float LAB[], TSharedCoordinates& tc) const
{
	assert((colorIndex >= 0) && (colorIndex < m_maxVectorColor));

	LAB[1] = m_vectorColorBasis[colorIndex][0];
	LAB[2] = m_vectorColorBasis[colorIndex][1];
}//--- getAll -----------------------------------------------------------

inline void
CVectorColor::GetAll(int colorIndex, TSharedCoordinates& tc) const
{
	assert((colorIndex >= 0) && (colorIndex < m_maxVectorColor));

	tc.m_lab[1] = m_vectorColorBasis[colorIndex][0];
	tc.m_lab[2] = m_vectorColorBasis[colorIndex][1];
}//--- get -----------------------------------------------------------

#endif
