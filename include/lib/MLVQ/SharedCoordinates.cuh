#pragma once
#include <Optix7.hpp>
#include <CUDABuffer.hpp>
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>
namespace RayTracerFacility
{
	struct SharedCoordinates
	{
		// the values to be used for interpolation in beta coordinate
		CudaBuffer m_betaAnglesBuffer;
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

		SharedCoordinates() {}
		SharedCoordinates(const bool& useCosBeta, const int& LengthOfSlice, std::vector<float>& betaAngles) {
			m_useCosBeta = useCosBeta;
			m_lengthOfSlice = LengthOfSlice;
			m_betaAnglesBuffer.Upload(betaAngles);
			m_betaAngles = reinterpret_cast<float*>(m_betaAnglesBuffer.DevicePointer());
			m_hdrFlag = false;
		}
		// Here we set the structure for particular angle beta
		__device__
			void SetForAngleBetaDeg(float beta)
		{
			assert((beta >= -90.f) && (beta <= 90.f));
			m_beta = beta;
			if (m_useCosBeta) {
				// The angles are quantized uniformly in sin of beta
				const float betaRad = beta * glm::pi<float>() / 180.f;
				const float sinBeta = sin(betaRad);
				m_iBeta = (int)(floor((sinBeta + 1.0f) / 2.0f * (m_lengthOfSlice - 1)));
				if (m_iBeta == m_lengthOfSlice - 1)
					m_iBeta = m_lengthOfSlice - 2; // beta = 90.f

				  //    assert( (iBeta >= 0) && (iBeta < LengthOfSlice-1) );

				  //	if (!((beta >= betaAngles[iBeta]) && (beta <= betaAngles[iBeta+1]))) {
				  //	  fprintf(stderr,"beta=%f ibeta=%f\n", beta, iBeta);
				  //	}

				  //	assert( (beta >= betaAngles[iBeta]) && (beta <= betaAngles[iBeta+1]) );

				  // Either we can interpolate in angles
				m_wBeta = (beta - m_betaAngles[m_iBeta]) / (m_betaAngles[m_iBeta + 1] - m_betaAngles[m_iBeta]);
				//assert( m_wBeta >= 0.f && m_wBeta <= 1.0f );
			}
			else {
				// The angles are quantized uniformly in degrees
				const float stepBeta = 180.f / (m_lengthOfSlice - 1);
				m_iBeta = (int)(floor((beta + 90.f) / stepBeta));
				if (m_iBeta > m_lengthOfSlice - 2)
					m_iBeta = m_lengthOfSlice - 2;
				//  printf("%d %f %f\n",i,beta/stepBeta);
				assert((m_iBeta >= 0) && (m_iBeta < m_lengthOfSlice - 1));
				m_wBeta = (beta + 90.f - m_iBeta * stepBeta) / stepBeta;
				assert((m_wBeta >= -1e-5f) && (m_wBeta <= 1.f));
				if (m_wBeta < 0.f)
					m_wBeta = 0.f;
			}
		}

		// Here we set the structure for particular angle alpha
		__device__
			void SetForAngleAlphaDeg(float alpha)
		{
			//assert((alpha >= -90.f) && (alpha <= 90.f));

			this->m_alpha = alpha;
			m_iAlpha = (int)floor((90.f + alpha) / m_stepAlpha);
			if (m_iAlpha > m_slicesPerHemi - 2)
				m_iAlpha = m_slicesPerHemi - 2;
			//assert((m_iAlpha >= 0) && (m_iAlpha <= m_slicesPerHemi - 1));
			m_wAlpha = (alpha + 90.f - m_iAlpha * m_stepAlpha) / m_stepAlpha;
			//assert((m_wAlpha >= -1e-5) && (m_wAlpha <= 1.f));
			if (m_wAlpha < 0.f)
				m_wAlpha = 0.f;
		}
	};
	__device__

		inline void ConvertThetaPhiToBetaAlpha(const float theta, const float phi,
			float& beta, float& alpha,
			const SharedCoordinates& tc)
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