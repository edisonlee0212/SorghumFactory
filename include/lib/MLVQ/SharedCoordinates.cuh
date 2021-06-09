#pragma once
#include <Optix7.hpp>
#include <glm/glm.hpp>
namespace RayTracerFacility
{
	struct TSharedCoordinates
	{
		// the values to be used for interpolation in beta coordinate
		float* m_betaAngles; // the sequence of values used
		int m_lengthOfSlice;

		// false ... use uniform distribution in Beta
		// true ... use uniform distribution in cos(Beta)
		bool m_useCosBeta;

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

		TSharedCoordinates() {}
		TSharedCoordinates(bool useCosBeta, int LengthOfSlice, float betaAnglesVals[]) {
			this->m_useCosBeta = useCosBeta;
			this->m_lengthOfSlice = LengthOfSlice;
			this->m_betaAngles = new float[LengthOfSlice];
			//assert(this->m_betaAngles);
			for (int i = 0; i < LengthOfSlice; i++)
				m_betaAngles[i] = betaAnglesVals[i];
			m_hdrFlag = false;
		}
		// Here we set the structure for particular angle beta
		__device__
			void SetForAngleBetaDeg(float beta);

		// Here we set the structure for particular angle alpha
		__device__
			void SetForAngleAlphaDeg(float alpha);
	};
	__device__

	inline void ConvertThetaPhiToBetaAlpha(const float theta, const float phi,
	                                       float& beta, float& alpha,
	                                       const TSharedCoordinates& tc)
	{
		if (tc.m_codeBtfFlag) {
			const float x = cos(phi - tc.m_phi) * sin(theta);
			const float y = sin(phi - tc.m_phi) * sin(theta);
			//float z = cos(thetaI);

			beta = asin(glm::clamp(y, -1.0f, 1.0f));
			const float cosBeta = cos(beta);

			if (cosBeta < 0.00001f)
			{
				alpha = 0.0f;
				return;
			}
			const float tmp = glm::clamp(-x / cosBeta, -1.0f, 1.0f);
			alpha = asin(tmp);
			return;
		}

		// This is three dimensional vector
		float xyz[3];

		// Here we convert the angles to 3D vector
		xyz[0] = cos(phi) * sin(theta);
		xyz[1] = sin(phi) * sin(theta);
		xyz[2] = cos(theta);

		// Here we convert 3D vector to alpha-beta parametrization over hemisphere
		beta = asin(xyz[0]);
		float aux = xyz[1] / cos(beta);
		if (aux < -1.f) aux = -1.f;
		if (aux > 1.f) aux = 1.f;
		assert((aux >= -1) && (aux <= 1));
		alpha = asin(aux);
	}
}