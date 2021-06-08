/*!**************************************************************************
\file    PDF2D.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  2D PDF database
******************************************************************************/

#ifndef PDF2Da_c
#define PDF2Da_c

#include <SharCoors.hpp>
#include <PDF2D.hpp>
#include <CIELab.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
//#################################################################################
//! \brief PDF2D - database of indices on 1D slices, norm. params., shifts, colours
//#################################################################################

// -----CPDF2DSeparate---------------------------------------------------
class CPDF2DSeparate :
	public CPDF2D
{
	// ===========================================================================
	// Here is the class representing only colors in 2D (alpha-beta parametrization)
	class CPDF2DColor {
		// the number of allocated 2D functions to be stored
		int m_maxPdf2D;
		// the used number of 2D functions
		int m_numOfPdf2D;
		// length of index slice
		int m_lengthOfSlice;
		// the number of indices in parameter alpha
		int m_slicesPerHemisphere;
		// the size of the data entry to be used here during restoration
		int m_size2D;
		// the database of indices of color 1D functions 
		CIndexAB* m_iab;

		// Here are the indices to CIndexAB class
		int** m_pdf2DColors;

		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		// check & reallocation of database if needed
		void Reallocate();
	public:
		CPDF2DColor(int maxPdf2D, int lengthOfSlice, int slicesPerHemisphere,
			CIndexAB* iab,
			float** restoredValuesOpt);
		~CPDF2DColor();
		void DeleteData();

		// Returns the number of 2D functions for color
		int GetNumOfPdf2DColor() const;

		// get a single value of A and B, only value A and B, posLAB=1 ... A, posLAB=2 ... B
		inline float Get(int pdf2DIndex, int posAlpha, int posBeta,
			int posLab, TSharedCoordinates& tc) const;
		inline void GetAll(int pdf2DIndex, int posAlpha, int posBeta,
			float lab[], TSharedCoordinates& tc) const;
		inline void GetAll(int pdf2DIndex, TSharedCoordinates& tc) const;

		// get single value for arbitrary angles alpha,beta and slice index, only values A and B
		void GetVal(int pdf2DIndex, float alpha, float beta,
			float lab[], TSharedCoordinates& tc) const;
		// Here alpha and beta is specified by 'tc', only values A and B
		void GetVal(int pdf2DIndex, float lab[], TSharedCoordinates& tc) const;
		bool GetValShepard(int pdf2DIndex, int ii, float sumDist2,
			float lab[], TSharedCoordinates& tc) const;

		//return memory in bytes required by the data representation
		int GetMemory() const;
		int GetMemoryQ() const;

		// the number of 2D luminance functions
		int Size() const { return m_numOfPdf2D; }

		int Load(char* prefix, int mlf, int maxIndexColor, int algPut);
	};

	// ===========================================================================
	// Here is the class representing only colors in 2D (alpha-beta parametrization)
	class CPDF2DLuminance {
		// the number of allocated 2D functions to be stored
		int m_maxPdf2D;
		// the used number of 2D functions
		int m_numOfPdf2D;
		// length of index slice
		int m_lengthOfSlice;
		// the number of indices in parameter alpha
		int m_slicesPerHemisphere;
		// the size of the data entry to be used here during restoration
		int m_size2D;
		// the database of 1D functions over luminance
		CPDF1D* m_pdf1;

		// Here are the indices to PDF1D class
		int** m_pdf2DSlices;
		// Here are the scale to PDF1D class, PDF1 functions are multiplied by that
		float** m_pdf2DScale;
		// This is optional, not required for rendering, except importance sampling
		float* m_pdf2DNorm;
		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		// check & reallocation of database if needed
		void Reallocation();

		enum { MaxDeltaCNT = 64 };
		float m_deltaArray[MaxDeltaCNT]; // the array of floats to be used for importance sampling

	public:
		CPDF2DLuminance(int maxPdf2D, int slicesPerHemisphere, CPDF1D* pdf1,
			int metric, float** restoredValuesOpt);
		~CPDF2DLuminance();
		void DeleteData();
		// Returns the number of 2D functions for luminance
		int GetNumOfPdf2DLuminance() const;

		// Restores the whole PDF2D function for luminance given the index
		int RestoreLum(float** restoredValues, int pdf2DIndex,
			TSharedCoordinates& tc);

		// get a single value of A and B, only luminance
		inline float Get(int pdf2DIndex, int posAlpha, int posBeta,
			TSharedCoordinates& tc) const;
		inline void GetAll(int pdf2DIndex, TSharedCoordinates& tc) const;

		// get single value for arbitrary angles alpha,beta and slice index, only luminances
		void GetVal(int pdf2DIndex, float alpha, float beta, float lab[],
			TSharedCoordinates& tc) const;
		// Here alpha and beta is specified by 'tc', only luminance
		void GetVal(int pdf2DIndex, float lab[], TSharedCoordinates& tc) const;
		void GetValShepard(int pdf2DIndex, int ii, float scale, float maxDist2, float lab[],
			TSharedCoordinates& tc) const;

		//return memory in bytes required by the data representation
		int GetMemory() const;
		int GetMemoryQ() const;

		int Load(char* prefix, int mlf, int maxPdf1D, int algPut);

		// compute the albedo for fixed viewer direction
		void GetViewerAlbedoDeg(int pdf2DIndex, float RGB[],
			float& normPar, TSharedCoordinates& tc);

		// compute importance sampling, returning the direction of incoming light
		int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
			int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
			float q0, float q1, float& illuminationTheta, float& illuminationPhi,
			TSharedCoordinates& tc);

		// compute importance sampling, returning the direction of incoming light
		// preparing 'cntRays' rays
		int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
			int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
			int cntRays, float q0Q1[], float illuminationThetaPhi[],
			TSharedCoordinates& tc);
		// the number of 2D luminance functions
		int Size() const { return m_numOfPdf2D; }
	};

	// Here are the instances of color and luminance 2D function database
	CPDF2DColor* m_color;
	CPDF2DLuminance* m_luminance;
	// Here are the indices of luminances + color 2D functions
	// index [][0] is luminance, index [][1] is color
	int** m_indexLuminanceColor;
	// The size of entries for 2D BTF slice
	int m_size2D;

	// These are the values, which are already normalized
	float** m_normValues;

	// normalise 2D PDF and returns normalization value
	float NormalPdf2D(const float* lfunc);
	float NormalPdf2DRgb(const float* lfunc);

	// check & reallocation of database if needed
	void Reallocate();

	// Here we return the normalization factor for existing 2D luminance function
	float GetNormalFactor(int pdf2DIndex) const;
	// Here we need the buffer to restore the values
	void InitLocalBuffer();

public:
	// constructor, allocates maximum number of 2D PDFs, the number of slices per 2D PDF,
	// 1D PDFs database, 1D colour index slices database, metric used for comparison
	CPDF2DSeparate(int maxPdf2D, int slicesPerHemisphere, CPDF1D* pdf1, CIndexAB* iab,
		int metric, bool belowThresholdSearch, bool insertNewDataIntoCache);
	~CPDF2DSeparate() override;
	void DeleteData() override;
	void DeleteDataLuminance() const;
	void DeleteDataColor() const;
	// get a single value
	float Get(int pdf2DIndex, int posAlpha, int posBeta,
			  int posLab, TSharedCoordinates& tc) override;
	void GetAll(int pdf2DIndex, int posAlpha, int posBeta,
				float scale, float rgb[], TSharedCoordinates& tc) override;
	void GetAll(int pdf2DIndex, TSharedCoordinates& tc) override;
	// get single value for arbitrary angles alpha,beta and slice index
	void GetVal(int pdf2DIndex, float alpha, float beta,
				float rgb[], TSharedCoordinates& tc) const override;
	// Here alpha and beta is specified by 'tc'
	void GetVal(int pdf2DIndex, float rgb[], TSharedCoordinates& tc) const override;
	void GetValShepard(int pdf2DIndex, float scale,
					   float dist2, TSharedCoordinates& tc) const override;

	// Return memory in bytes required by the data representation, excluding
	// embedded functions for luminance and color 2D functions
	int GetMemory() const override;
	int GetMemoryQ() const override;

	int Load(char* prefix, int mlf, int maxPdf1D, int maxIndexColor, int algPut) override;

	int GetNumOfPdf2DLuminance() const;
	int GetMemoryLuminance() const;
	int GetMemoryLuminanceQ() const;

	int GetNumOfPdf2DColor() const;
	int GetMemoryColor() const;
	int GetMemoryColorQ() const;

	// compute the albedo for fixed viewer direction
	void GetViewerAlbedoDeg(int pdf2DIndex, float rgb[],
							float& normPar, TSharedCoordinates& tc) override;

	//! \brief computes importance sampling given the coordinates and [0-1]^2
	// random variables. It compute the normalized direction.
	int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
						   int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
						   float q0, float q1, float& illuminationTheta, float& illuminationPhi,
						   TSharedCoordinates& tc) override;
	int ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
						   int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
						   int cntRays, float q0Q1[], float illuminationThetaPhi[],
						   TSharedCoordinates& tc) override;
};//--- CPDF2DSeparate ---------------------------------------------------------

// get a single value of A and B
inline float
CPDF2DSeparate::CPDF2DColor::Get(const int pdf2DIndex, const int posAlpha, const int posBeta,
								 const int posLab, TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((posAlpha >= 0) && (posAlpha < m_slicesPerHemisphere));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));
	assert((posLab >= 1) && (posLab <= 2));

	// colours
	return m_iab->Get(m_pdf2DColors[pdf2DIndex][posAlpha], posBeta, posLab - 1, tc);  // colours a-b  
} // ----------------- CPDF2Dcolor::get ---------------------------------------

// get a single value of A and B
inline void
CPDF2DSeparate::CPDF2DColor::GetAll(const int pdf2DIndex, const int posAlpha, const int posBeta,
									float lab[], TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((posAlpha >= 0) && (posAlpha < m_slicesPerHemisphere));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));

	// colours
	m_iab->GetAll(m_pdf2DColors[pdf2DIndex][posAlpha], posBeta, lab, tc);  // colours a-b
} // ----------------- CPDF2DColor::get ---------------------------------------

// get a single value of A and B
inline void
CPDF2DSeparate::CPDF2DColor::GetAll(const int pdf2DIndex, TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((tc.m_indexAlpha >= 0) && (tc.m_indexAlpha < m_slicesPerHemisphere));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_iab->GetSliceLength()));

	// colors
	m_iab->GetAll(m_pdf2DColors[pdf2DIndex][tc.m_indexAlpha], tc);  // colors a-b
} // ----------------- CPDF2DColor::get ---------------------------------------

// get a single value of luminance
inline float
CPDF2DSeparate::CPDF2DLuminance::Get(const int pdf2DIndex, const int posAlpha, const int posBeta,
                                     TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((posAlpha >= 0) && (posAlpha < m_slicesPerHemisphere));
	assert((posBeta >= 0) && (posBeta < m_lengthOfSlice));

	// luminance
	return m_pdf2DScale[pdf2DIndex][posAlpha] *
		m_pdf1->Get(m_pdf2DSlices[pdf2DIndex][posAlpha], posBeta, tc);
} // ---------------------- CPDF2Dlum::get ------------------------------------

// get a single value of luminance
inline void
CPDF2DSeparate::CPDF2DLuminance::GetAll(const int pdf2DIndex, TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((tc.m_indexAlpha >= 0) && (tc.m_indexAlpha < m_slicesPerHemisphere));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_lengthOfSlice));

	// luminance
	tc.m_scale *= m_pdf2DScale[pdf2DIndex][tc.m_indexAlpha];
	m_pdf1->GetAll(m_pdf2DSlices[pdf2DIndex][tc.m_indexAlpha], tc);
} // ---------------------- CPDF2Dlum::get ------------------------------------

// get a single value
inline float
CPDF2DSeparate::Get(const int pdf2DIndex, const int posAlpha, const int posBeta,
                    const int posLab, TSharedCoordinates& tc)
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((posAlpha >= 0) && (posAlpha < m_numOfSlicesPerHemisphere));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));
	assert((posLab >= 0) && (posLab <= 2));

	if (posLab == 0)
		return m_luminance->Get(m_indexLuminanceColor[pdf2DIndex][0], posAlpha, posBeta, tc);

	return m_color->Get(m_indexLuminanceColor[pdf2DIndex][1], posAlpha, posBeta, posLab, tc);
} // ---------------------- get() ----------------------------------------

inline
void
CPDF2DSeparate::GetAll(const int pdf2DIndex, const int posAlpha, const int posBeta,
                       const float scale, float rgb[], TSharedCoordinates& tc)
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((posAlpha >= 0) && (posAlpha < m_numOfSlicesPerHemisphere));
	assert((posBeta >= 0) && (posBeta < m_iab->GetSliceLength()));

	float LAB[3];
	LAB[0] = scale * m_luminance->Get(m_indexLuminanceColor[pdf2DIndex][0], posAlpha, posBeta, tc);

	m_color->GetAll(m_indexLuminanceColor[pdf2DIndex][1], posAlpha, posBeta, LAB, tc);

	// Convert to RGB
	UserCmToRgb(LAB, rgb, tc);

} // ---------------------- getAll() ----------------------------------------

inline
void
CPDF2DSeparate::GetAll(const int pdf2DIndex, TSharedCoordinates& tc)
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((tc.m_indexAlpha >= 0) && (tc.m_indexAlpha < m_numOfSlicesPerHemisphere));
	assert((tc.m_indexBeta >= 0) && (tc.m_indexBeta < m_iab->GetSliceLength()));

	// get luminance
	m_luminance->GetAll(m_indexLuminanceColor[pdf2DIndex][0], tc);
	tc.m_lab[0] = tc.m_scale;

	// get color
	m_color->GetAll(m_indexLuminanceColor[pdf2DIndex][1], tc);

	// Convert to RGB
	UserCmToRgb(tc.m_lab, tc.m_rgb, tc);

} // ---------------------- getAll() ----------------------------------------

#endif
