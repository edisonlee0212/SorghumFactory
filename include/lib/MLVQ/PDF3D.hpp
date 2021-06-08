/*!**************************************************************************
\file    PDF3D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  3D PDF indices database
******************************************************************************/

#ifndef PDF3D_c
#define PDF3D_c
#include <IndexAB.hpp>
#include <PDF1D.hpp>
#include <PDF2D.hpp>

//#################################################################################
//! \brief PDF3D - database of indices on 3D slices, norm. params.
//#################################################################################

class CPDF3D {
private:
	// the number of allocated 3D functions to be stored
	int m_maxPdf3D;
	// the used number of 3D functions
	int m_numOfPdf3D;
	// the number of slices per theta (=2D functions) to represent one 3D function
	int m_slicesPerTheta;
	// angle theta quantization step
	float m_stepTheta;
	// the size of the data entry to be used here during restoration
	int m_size3D;

	// These are the data allocated maxPDF2D times, serving to represent the function
	int** m_pdf3Dslices;
	float** m_pdf3Dscale;

	// the database of 1D functions
	CPDF1D* m_pdf1;
	// the database of indices of colour functions
	CIndexAB* m_iab;
	// the database of 2D functions to which we point in the array PDF3Dslices
	CPDF2D* m_pdf2;

	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database
public:
	CPDF3D(int maxPDF3D, int SlicesPerTheta, CPDF1D* PDF1, CIndexAB* IAB, CPDF2D* PDF2,
		int metric);
	~CPDF3D();
	void DeleteData();

	// check & reallocation of database if needed
	void Reallocate();

	// get a single value
	inline float Get(int PDF3Dindex, int indexTheta, int posAlpha, int posBeta,
		int posLAB, TSharedCoordinates& tc) const;
	inline void GetAll(int PDF3Dindex, int indexTheta, int posAlpha, int posBeta,
		float scale, float RGB[], TSharedCoordinates& tc) const;
	inline void GetAll(int PDF3Dindex, TSharedCoordinates& tc) const;

	// get single value for arbitrary angles theta,alpha,beta and slice index
	void GetVal(int pdf3DIndex, float viewTheta, float alpha, float beta, float rgb[],
		TSharedCoordinates& tc) const;
	// here theta_v, alpha, and beta specified by tc
	void GetVal(int pdf3DIndex, float rgb[], TSharedCoordinates& tc) const;
	void GetValShepard(int pdf3DIndex, float scale, float dist2,
		TSharedCoordinates& tc) const;

	// S_5 .. the number of 3D PDF functions in the 3D database
	int GetNoOfPdf3D() const { return m_numOfPdf3D; }

	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;

	int GetSlicesPerTheta() const { return m_slicesPerTheta; }
	float GetStepTheta() const { return m_stepTheta; }

	int Load(char* prefix, int MLF, int maxPDF2, int algPut);

	void GetViewerAlbedoDeg(int PDF3Dindex, float theta_v, float RGB[], float& normParV,
		TSharedCoordinates& tc);

	//! \brief computes importance sampling given the coordinates and [0-1]^2
	// random variables. It compute the normalized direction.
	int ImportanceSamplingDeg(int PDF3Dindex1, float w1, int PDF3Dindex2, float w2,
		float q0, float q1, float& theta_i, float& phi_i,
		TSharedCoordinates& tc);
	int ImportanceSamplingDeg(int pdf3DIndex1, float w1, int pdf3DIndex2, float w2,
		int cntRays, float q0Q1[], float illuminationThetaPhi[],
		TSharedCoordinates& tc) const;
};//--- CPDF3D ---------------------------------------------------------

// get a single value
inline float
CPDF3D::Get(int PDF3Dindex, int indexTheta, int posAlpha, int posBeta,
	int posLAB, TSharedCoordinates& tc) const
{
	assert((PDF3Dindex >= 0) && (PDF3Dindex < m_maxPdf3D));
	assert((indexTheta >= 0) && (indexTheta < m_slicesPerTheta));
	assert((posAlpha >= 0) && (posAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));
	assert((posLAB >= 0) && (posLAB <= 2));

	if (posLAB == 0)
		return m_pdf3Dscale[PDF3Dindex][indexTheta] *
		m_pdf2->Get(m_pdf3Dslices[PDF3Dindex][indexTheta], posAlpha, posBeta, posLAB, tc);
	return m_pdf2->Get(m_pdf3Dslices[PDF3Dindex][indexTheta], posAlpha, posBeta, posLAB, tc);
}//--- get -------------------------------------------------------------------

// get a single value
inline void
CPDF3D::GetAll(int PDF3Dindex, int indexTheta, int posAlpha, int posBeta,
	float scale, float RGB[], TSharedCoordinates& tc) const
{
	assert((PDF3Dindex >= 0) && (PDF3Dindex < m_maxPdf3D));
	assert((indexTheta >= 0) && (indexTheta < m_slicesPerTheta));
	assert((posAlpha >= 0) && (posAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));

	scale *= m_pdf3Dscale[PDF3Dindex][indexTheta];
	m_pdf2->GetAll(m_pdf3Dslices[PDF3Dindex][indexTheta], posAlpha, posBeta, scale, RGB, tc);
}//--- getAll -------------------------------------------------------------------

// get a single value
inline void
CPDF3D::GetAll(int PDF3Dindex, TSharedCoordinates& tc) const
{
	assert((PDF3Dindex >= 0) && (PDF3Dindex < m_maxPdf3D));
	assert((tc.m_indexTheta >= 0) && (tc.m_indexTheta < m_slicesPerTheta));
	assert((tc.m_indexAlpha >= 0) && (tc.m_indexAlpha < m_pdf2->GetSlicesPerHemisphere()));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_iab->GetSliceLength()));

	tc.m_scale *= m_pdf3Dscale[PDF3Dindex][tc.m_indexTheta];
	m_pdf2->GetAll(m_pdf3Dslices[PDF3Dindex][tc.m_indexTheta], tc);
}//--- getAll -------------------------------------------------------------------

#endif
