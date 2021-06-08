/*!**************************************************************************
\file    PDF2D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  2D PDF database
******************************************************************************/

#ifndef PDF2D_c
#define PDF2D_c
#include <IndexAB.hpp>
#include <PDF1D.hpp>
#include <SharCoors.hpp>

//#################################################################################
//! \brief PDF2D - database of indices on 1D slices, norm. params., shifts, colours
//#################################################################################

class CPDF2D
{
protected:
	// the number of allocated 2D functions to be stored
	int m_maxPdf2D;
	// the used number of 2D functions
	int m_numOfPdf2D;
	// the number of indices in parameter alpha
	int m_numOfSlicesPerHemisphere;
	// the size of the data entry to be used here during restoration
	int m_size2D;
	// the database of 1D functions over luminance
	CPDF1D* m_pdf1;
	// the database of indices of color functions
	CIndexAB* m_iab;
	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database
public:
	// constructor, allocates maximum number of 2D PDFs, the number of slices per 2D PDF,
	// 1D PDFs database, 1D colour index slices database, metric used for comparison
	CPDF2D(int maxPdf2D, int slicesPerHemisphere, CPDF1D* pdf1, CIndexAB* iab);

	virtual ~CPDF2D();
	virtual void DeleteData() = 0;

	int GetSlicesPerHemisphere() const { return m_numOfSlicesPerHemisphere; }

	//return memory in bytes required by the data representation
	virtual int GetMemory() const = 0;
	virtual int GetMemoryQ() const = 0;
	// S_4 .. the number of 2D PDF functions in the 2D database
	int GetNoOfPdf2D() const { return m_numOfPdf2D; }

	// get a single value
	virtual float Get(int pdf2DIndex, int posAlpha, int posBeta, int posLab,
		TSharedCoordinates& tc) = 0;
	virtual void GetAll(int pdf2DIndex, int posAlpha, int posBeta, float scale, float rgb[],
		TSharedCoordinates& tc) = 0;
	virtual void GetAll(int pdf2DIndex, TSharedCoordinates& tc) = 0;

	// get single value for arbitrary angles alpha,beta and slice index
	virtual void GetVal(int pdf2DIndex, float alpha, float beta, float rgb[],
		TSharedCoordinates& tc) const = 0;
	// Here alpha and beta is specified by 'tc'
	virtual void GetVal(int pdf2DIndex, float rgb[],
		TSharedCoordinates& tc) const = 0;
	virtual void GetValShepard(int pdf2DIndex, float scale, float dist2,
		TSharedCoordinates& tc) const = 0;

	virtual int Load(char* prefix, int mlf, int maxPdf1D, int maxIndexColor,
		int algPut) = 0;

	// compute the albedo for fixed viewer direction
	virtual void GetViewerAlbedoDeg(int pdf2DIndex, float rgb[],
		float& normPar, TSharedCoordinates& tc) = 0;

	//! \brief computes importance sampling given the coordinates and [0-1]^2
	// random variables. It computes the normalized direction accoring to q0,q1 and
	// shape of the function
	virtual int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
		int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
		float q0, float q1, float& illuminationTheta, float& illuminationPhi,
		TSharedCoordinates& tc) = 0;
	virtual int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
		int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
		int cntRays, float q0Q1[], float illuminationThetaPhi[],
		TSharedCoordinates& tc) = 0;
};//--- CPDF2Dabstract ---------------------------------------------------------

#endif
