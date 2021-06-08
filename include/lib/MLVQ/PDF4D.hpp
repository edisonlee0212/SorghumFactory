/*!**************************************************************************
\file    PDF4D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  4D PDF indices database
******************************************************************************/

#ifndef PDF4D_c
#define PDF4D_c

#include <IndexAB.hpp>
#include <PDF1D.hpp>
#include <PDF2D.hpp>
#include <PDF3D.hpp>

//#################################################################################
//! \brief PDF4D - database of indices on 4D slices, norm. params.
//#################################################################################

class CPDF4D {
private:
	// the number of allocated 4D functions to be stored
	int m_maxPdf4D;
	// the used number of 4D functions
	int m_numOfPdf4D;
	// the number of slices per phi (=3D functions) to represent one 4D function
	int m_slicesPerPhi;
	// angle phi quantization step
	float m_stepPhi;
	// the size of the data entry to be used here during restoration
	int m_size4D;

	// These are the data allocated maxPDF4D times, serving to represent the function
	int** m_pdf4DSlices;
	float** m_pdf4DScale;

	// the database of 1D functions
	CPDF1D* m_pdf1;
	// the database of indices of colour functions
	CIndexAB* m_iab;
	// the database of 2D functions to which we point in the array PDF3Dslices
	CPDF2D* m_pdf2;
	// the database of 3D functions to which we point in the array PDF4Dslices
	CPDF3D* m_pdf3;

	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database  
public:
	CPDF4D(int maxPDF4D, int SlicesPerPhi, CPDF1D* PDF1, CIndexAB* IAB, CPDF2D* PDF2,
		CPDF3D* PDF3, int metric);

	~CPDF4D();
	void DeleteData();

	// check & reallocation of database if needed
	void Reallocate();

	// get a single value
	inline float Get(int PDF4Dindex, int indexPhi, int indexTheta, int posAlpha, int posBeta,
		int posLAB, TSharedCoordinates& tc) const;

	// Shepard interpolation
	inline void GetAll(int PDF4Dindex, int indexPhi, int indexTheta, int posAlpha, int posBeta,
		float RGB[], TSharedCoordinates& tc) const;
	// Here index specified in tc
	inline void GetAll(int PDF4Dindex, TSharedCoordinates& tc) const;

	// get single value for arbitrary angles phi,theta,alpha,beta and slice index
	void GetVal(int PDF4Dindex, float phi_v, float theta_v, float alpha,
		float beta, float RGB[], TSharedCoordinates& tc) const;
	// here phi_v, theta_v, alpha, and beta specified by tc
	void GetVal(int pdf4DIndex, float rgb[], TSharedCoordinates& tc) const;
	// Shepard based interpolation
	void GetValShepard(int PDF4Dindex, TSharedCoordinates& tc) const;

	int GetNoOfPdf4D() const;// S_6 .. the number of 4D PDF functions in the 4D database

	int GetSlicesPerPhi() const;
	float GetStepPhi() const { return m_stepPhi; }

	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;

	int Load(char* prefix, int MLF, int maxPDF3D, int algPut);

	//! \brief computes albedo for fixed viewer direction
	void GetViewerAlbedoDeg(int PDF4Dindex, float theta_v, float phi_v,
		float RGB[], TSharedCoordinates& tc);

	//! \brief computes importance sampling given the coordinates and [0-1]^2
	// random variables. It compute the normalized direction.
	int ImportanceSamplingDeg(int PDF4Dindex, float q0, float q1, float& theta_i, float& phi_i,
		TSharedCoordinates& tc);
	int ImportanceSamplingDeg(int pdf4Index, int cntRays, float q0Q1[], float illuminationThetaPhi[],
		TSharedCoordinates& tc) const;
};//--- CPDF4D ---------------------------------------------------------

// get a single value
inline float
CPDF4D::Get(int PDF4Dindex, int indexPhi, int indexTheta,
	int posAlpha, int posBeta, int posLAB, TSharedCoordinates& tc) const
{
	assert((PDF4Dindex >= 0) && (PDF4Dindex < m_maxPdf4D));
	assert((indexPhi >= 0) && (indexPhi < m_slicesPerPhi));
	assert((indexTheta >= 0) && (indexTheta < m_pdf3->GetSlicesPerTheta()));
	assert((posAlpha >= 0) && (posAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));
	assert((posLAB >= 0) && (posLAB <= 2));

	if (posLAB == 0)
		return m_pdf4DScale[PDF4Dindex][indexPhi] *
		m_pdf3->Get(m_pdf4DSlices[PDF4Dindex][indexPhi], indexTheta, posAlpha, posBeta,
			posLAB, tc);
	return m_pdf3->Get(m_pdf4DSlices[PDF4Dindex][indexPhi], indexTheta, posAlpha, posBeta,
		posLAB, tc);
}//--- get -------------------------------------------------------------------

// get a single value in RGB space
inline void
CPDF4D::GetAll(int PDF4Dindex, int indexPhi, int indexTheta,
	int posAlpha, int posBeta, float RGB[], TSharedCoordinates& tc) const
{
	assert((PDF4Dindex >= 0) && (PDF4Dindex < m_maxPdf4D));
	assert((indexPhi >= 0) && (indexPhi < m_slicesPerPhi));
	assert((indexTheta >= 0) && (indexTheta < m_pdf3->GetSlicesPerTheta()));
	assert((posAlpha >= 0) && (posAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));

	float scale = m_pdf4DScale[PDF4Dindex][indexPhi];
	m_pdf3->GetAll(m_pdf4DSlices[PDF4Dindex][indexPhi], indexTheta, posAlpha, posBeta,
		scale, RGB, tc);

	return;
}//--- getAll -------------------------------------------------------------------

// get a single value in RGB space
inline void
CPDF4D::GetAll(int PDF4Dindex, TSharedCoordinates& tc) const
{
	assert((PDF4Dindex >= 0) && (PDF4Dindex < m_maxPdf4D));
	assert((tc.m_indexPhi >= 0) && (tc.m_indexPhi < m_slicesPerPhi));
	assert((tc.m_indexTheta >= 0) && (tc.m_indexTheta < m_pdf3->GetSlicesPerTheta()));
	assert((tc.m_indexAlpha >= 0) && (tc.m_indexAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_iab->GetSliceLength()));

	tc.m_scale = m_pdf4DScale[PDF4Dindex][tc.m_indexPhi];
	m_pdf3->GetAll(m_pdf4DSlices[PDF4Dindex][tc.m_indexPhi], tc);

	return;
}//--- getAll -------------------------------------------------------------------

#endif
