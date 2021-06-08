/*!**************************************************************************
\file    SharCoors.h
\author  Vlastimil Havran
\date    15/11/2006
\version 1.01

  The header file for the:
	BTFBASE project with V. Havran (MPI)

*****************************************************************************/
#ifndef SharCoors_c
#define SharCoors_c

#include <cassert>

// If defines, shared coordinates are shared
#define BTFSC 1

inline float
Clamp(const float a, const float min, const float max)
{
	if (a < min)
		return min;
	if (a > max)
		return max;
	return a;
}

/// This class represents the values and indices to be used
// for the interpolation, when we retrieve a value
class TSharedCoordinates
{
public:
	TSharedCoordinates() {}
	TSharedCoordinates(bool useCosBeta, int LengthOfSlice, float betaAnglesVals[]) {
		this->m_useCosBeta = useCosBeta;
		this->m_lengthOfSlice = LengthOfSlice;
		this->m_betaAngles = new float[LengthOfSlice];
		assert(this->m_betaAngles);
		for (int i = 0; i < LengthOfSlice; i++)
			m_betaAngles[i] = betaAnglesVals[i];
		m_hdrFlag = false;
	}

	~TSharedCoordinates() {
		// $$ do not delete as the array is reused...
		//	delete [] betaAngles; betaAngles = 0;
	}

	// the values to be used for interpolation in beta coordinate
	float* m_betaAngles; // the sequence of values used
	int m_lengthOfSlice;

	// false ... use uniform distribution in Beta
	// true ... use uniform distribution in cos(Beta)
	bool m_useCosBeta;
	// Here we set the structure for particular angle beta
	void SetForAngleBetaDeg(float beta);

	// Here we set the structure for particular angle alpha
	void SetForAngleAlphaDeg(float alpha);

	// Here we compute index directly to the variables
	void ComputeIndexForAngleBetaDeg(float beta, int& i, float& w) const;
	// We return the angle in degrees given the index of slice
	float GetAngleBetaByIndex(int i) const
	{
		assert(i >= 0); assert(i < m_lengthOfSlice);
		return m_betaAngles[i];
	}
	// We given the index of the first angle + weight compute beta angle
	float ComputeAngleBeta(int i, float w);

	// We given the index of the first angle + weight compute alpha angle
	float ComputeAngleAlpha(int i, float w) const
	{
		assert(i >= 0);
		assert(i < m_slicesPerHemi);
		if (i == m_slicesPerHemi - 1) {
			assert(w == 0.f);
			return 90.f;
		}
		return -90.f + (float)(i + w) * m_stepAlpha;
	}

	float m_stepAlpha;
	int m_slicesPerHemi;

	float m_stepTheta;
	int m_slicesPerTheta;

	float m_stepPhi;
	int m_slicesPerPhi;

	// the BTF single point coordinates in degrees
	float m_beta;  //1D
	float m_alpha; //2D
	float m_theta; //3D
	float m_phi;   //4D

	// interpolation values for PDF1D
	int m_iBeta;
	float m_wBeta;
	float m_wMinBeta2;

	// interpolation values for PDF2D
	int m_iAlpha;
	float m_wAlpha;
	float m_wMinAlpha2;

	// interpolation values for PDF3D
	int m_iTheta;
	float m_wTheta;
	float m_wMinTheta2;

	// interpolation values for PDF4D
	int m_iPhi;
	float m_wPhi;

	// Here are the results for Shepard based interpolation
	float m_maxDist;
	float m_maxDist2;
	float m_rgb[4], m_lab[4];
	float m_sumWeight;
	int   m_countWeight;

	// for indexing for Shepard interpolation
	int m_indexPhi, m_indexTheta, m_indexBeta, m_indexAlpha;
	float m_scale;

	bool m_hdrFlag;
	bool m_codeBtfFlag;
};

// Here is the conversion of the hemispherical coordinates
// Angles are in radians
void
ConvertThetaPhiToBetaAlpha(float theta, float phi,
	float& beta, float& alpha,
	const TSharedCoordinates& tc);

// Angles are in radians
void
ConvertBetaAlphaToThetaPhi(float beta, float alpha,
	float& theta, float& phi);

// Directon to angles
void
ConvertDirectionToThetaPhi(float dirx, float diry, float dirz,
	float& theta, float& phi);

// Here is the conversion from angles to unit vector
// Angles are in radians
void
ConvertBetaAlphaToXyz(float beta, float alpha, float xyz[]);

// Angles are in radians
void
ConvertThetaPhiToXyz(float theta, float phi, float xyz[]);

#endif // SharCoors_c
