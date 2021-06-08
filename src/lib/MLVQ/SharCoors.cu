/*!**************************************************************************
\file    SharCoors.h
\author  Vlastimil Havran
\date    15/11/2006
\version 1.01

  The header file for the:
	BTFBASE project with V. Havran (MPI)

*****************************************************************************/

#include <cmath>
#include <cassert>

#include <SharCoors.hpp>
#include <AuxFuncs.hpp>

// The method that converts the directional vector on the hemisphere to the 
// angle theta (from zenith) and phi angle (the azimuth)
void
ConvertDirectionToThetaPhi(float dirx, float diry, float dirz,
	float& theta,
	float& phi)
{
	float mag = sqrt(dirx * dirx + diry * diry + dirz * dirz);
	dirx /= mag;
	diry /= mag;
	dirz /= mag;

	// Standard conversion
	theta = acos(dirz);
	phi = atan2(diry, dirx);

	if (phi < 0.f)
		phi += 2 * PI;

	assert((phi >= 0.f) && (phi <= 2.f * PI));
	return;
}

void
ConvertThetaPhiToBetaAlpha(const float theta, const float phi,
	float& beta, float& alpha,
	const TSharedCoordinates& tc)
{
	if (tc.m_codeBtfFlag) {
		const float x = cos(phi - tc.m_phi) * sin(theta);
		const float y = sin(phi - tc.m_phi) * sin(theta);
		//float z = cos(thetaI);

		beta = asin(Clamp(y, -1.0f, 1.0f));
		const float cosBeta = cos(beta);

		if (cosBeta < 0.00001f)
		{
			alpha = 0.0f;
			return;
		}
		const float tmp = Clamp(-x / cosBeta, -1.0f, 1.0f);
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

void
ConvertBetaAlphaToThetaPhi(float beta, float alpha,
	float& theta, float& phi)
{
	// first convert to 3D vector
	float xyz[3];
	xyz[0] = sin(beta);
	xyz[1] = sin(alpha) * cos(beta);
	xyz[2] = cos(alpha) * cos(beta);
	// then convert to theta, phi
	theta = acos(xyz[2]);
	float s = sin(theta);
	if (s < 1e-5) s = 1e-5;
	xyz[0] /= s;
	xyz[1] /= s;
	phi = atan2(xyz[1], xyz[0]);
	while (phi < 0.f)
		phi += 2.0f * PI;
}

void
ConvertBetaAlphaToXyz(float beta, float alpha, float xyz[])
{
	xyz[0] = sin(beta);
	xyz[1] = sin(alpha) * cos(beta);
	xyz[2] = cos(alpha) * cos(beta);
}

void
ConvertThetaPhiToXyz(float theta, float phi, float xyz[])
{
	// Here we convert the angles to 3D vector
	xyz[0] = cos(phi) * sin(theta);
	xyz[1] = sin(phi) * sin(theta);
	xyz[2] = cos(theta);
}

void
TSharedCoordinates::SetForAngleBetaDeg(const float beta)
{
	assert((beta >= -90.f) && (beta <= 90.f));
	this->m_beta = beta;
	if (m_useCosBeta) {
		// The angles are quantized uniformly in sin of beta
		const float betaRad = beta * PI / 180.f;
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
		//    assert( (wBeta >= 0.f)&&(wBeta <= 1.0f) );
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
void
TSharedCoordinates::SetForAngleAlphaDeg(const float alpha)
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

// Here we compute index directly to the variables
void
TSharedCoordinates::ComputeIndexForAngleBetaDeg(
	float betaUser, // input
	int& indexBeta, float& weightBeta) const // output
{
	assert((betaUser >= -90.f) && (betaUser <= 90.f));
	if (m_useCosBeta) {
		// The angles are quantized uniformly in sin of beta
		float betaRad = betaUser * PI / 180.f;
		float sinBeta = sin(betaRad);
		indexBeta = (int)(floor((sinBeta + 1.0f) / 2.0f * (m_lengthOfSlice - 1)));
		assert((indexBeta >= 0) && (indexBeta < m_lengthOfSlice - 1));
		assert((betaUser >= m_betaAngles[indexBeta]) && (betaUser < m_betaAngles[indexBeta + 1]));
		// Either we can interpolate in angles
		weightBeta = (betaUser - m_betaAngles[indexBeta]);
		weightBeta /= (m_betaAngles[indexBeta + 1] - m_betaAngles[indexBeta]);
		assert((weightBeta >= 0.f) && (weightBeta <= 1.0f));
	}
	else {
		// The angles are quantized uniformly in degrees
		float stepBeta = 180.f / (m_lengthOfSlice - 1);
		indexBeta = (int)(floor((betaUser + 90.f) / stepBeta));
		if (indexBeta > m_lengthOfSlice - 2)
			indexBeta = m_lengthOfSlice - 2;
		//  printf("%d %f %f\n",i,beta/stepBeta);
		assert((indexBeta >= 0) && (indexBeta < m_lengthOfSlice - 1));
		weightBeta = (m_beta + 90.f - indexBeta * stepBeta) / stepBeta;
		assert((weightBeta >= -1e-5f) && (weightBeta <= 1.f));
		if (weightBeta < 0.f)
			weightBeta = 0.f;
	}

	return;
}

// Here we compute index directly to the variables
float
TSharedCoordinates::ComputeAngleBeta(int indexBeta, float w)
{
	float beta;
	assert((indexBeta >= 0) && (indexBeta <= m_lengthOfSlice));

	if (indexBeta == m_lengthOfSlice) {
		assert(w == 0.f);
		return 90.f;
	}

	assert((w >= 0.f) && (w <= 1.0f));
	if (m_useCosBeta) {
		// here we have non-uniform quantization
		// Either we can interpolate in angles
		beta = m_betaAngles[indexBeta] + (m_betaAngles[indexBeta + 1] - m_betaAngles[indexBeta]) * w;
	}
	else {
		// The angles are quantized uniformly in degrees
		float stepBeta = 180.f / (m_lengthOfSlice - 1);
		beta = -90.f + stepBeta * ((float)indexBeta + w);
		assert((beta >= -90.f) && (beta <= 90.f));
	}

	return beta;
}
