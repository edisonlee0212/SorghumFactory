#include <SharedCoordinates.cuh>
#include <glm/ext/scalar_constants.hpp>

void RayTracerFacility::TSharedCoordinates::SetForAngleBetaDeg(float beta)
{
	assert((beta >= -90.f) && (beta <= 90.f));
	this->m_beta = beta;
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

void RayTracerFacility::TSharedCoordinates::SetForAngleAlphaDeg(float alpha)
{
	assert((alpha >= -90.f) && (alpha <= 90.f));

	this->m_alpha = alpha;
	m_iAlpha = (int)floor((90.f + alpha) / m_stepAlpha);
	if (m_iAlpha > m_slicesPerHemi - 2)
		m_iAlpha = m_slicesPerHemi - 2;
	assert((m_iAlpha >= 0) && (m_iAlpha <= m_slicesPerHemi - 1));
	m_wAlpha = (alpha + 90.f - m_iAlpha * m_stepAlpha) / m_stepAlpha;
	assert((m_wAlpha >= -1e-5) && (m_wAlpha <= 1.f));
	if (m_wAlpha < 0.f)
		m_wAlpha = 0.f;
}
