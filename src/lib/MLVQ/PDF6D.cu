/*!**************************************************************************
\file    PDF6D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  6D PDF indices database
******************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <PDF4D.hpp>
#include <PDF6D.hpp>
#include <TBTFbase.hpp>

//#########################################################################
//######## CPDF6D #####################################################JF##
//#########################################################################

CPDF6D::CPDF6D(int nrows, int ncols, CPDF4D* PDF4
)
{
	this->m_pdf4 = PDF4;
	Allocate(nrows, ncols);
	// set offset
	this->m_rowsOffset = 0;
	this->m_colsOffset = 0;
}//--- CPDF6 -----------------------------------------------------------------

// This is required for SSIM data precomputation - setting the dimensions
void
CPDF6D::SetSizes(int nSlicesPerPhi, int nSlicesPerTheta, int nSlicesPerHemi,
	int nLengthOfSlice, int nncolour)
{
	this->m_slicesPerPhi = nSlicesPerPhi;
	this->m_slicesPerTheta = nSlicesPerTheta;
	this->m_slicesPerHemisphere = nSlicesPerHemi;
	this->m_lengthOfSlice = nLengthOfSlice;
	this->m_numOfColors = nncolour;
}

CPDF6D::~CPDF6D()
{
	CleanObject();
}//--- ~CPDF6 ----------------------------------------------------------------

void
CPDF6D::Allocate(int nrows, int ncols)
{
	this->m_numOfRows = nrows;
	this->m_numOfCols = ncols;
	m_pdf6DSlices = IAllocation2(0, nrows, 0, ncols);
	// Here we need to reset the array. The first valid
	// index of PDF4 is 1, 0 means that the pixel has not
	// been compressed yet !
	for (int irow = 0; irow < nrows; irow++)
		for (int jcol = 0; jcol < ncols; jcol++)
			m_pdf6DSlices[irow][jcol] = 0;

	m_pdf6DScale = Allocation2(0, nrows, 0, ncols);
}//--- alloc ------------------------------------------------------------------

void
CPDF6D::CleanObject()
{
	if (m_pdf6DSlices != NULL)
		IFree2(m_pdf6DSlices, 0, m_numOfRows, 0, m_numOfCols);
	if (m_pdf6DScale != NULL)
		Free2(m_pdf6DScale, 0, m_numOfRows, 0, m_numOfCols);
	m_numOfRows = 0;
	m_numOfCols = 0;
	m_rowsOffset = 0;
	m_colsOffset = 0;
	m_pdf6DSlices = NULL;
	m_pdf6DScale = NULL;
}//--- cleanObject -----------------------------------------------------------

// sets ofset of the first pixel to be addressed correctly
void
CPDF6D::SetOffset(int rows_offset, int cols_offset)
{
	this->m_rowsOffset = rows_offset;
	this->m_colsOffset = cols_offset;
}//--- setOffset ------------------------------------------------------------


// get a single value in LAB space
float
CPDF6D::Get(int y, int x, int indexPhi, int indexTheta,
	int posAlpha, int posBeta, int posLAB, TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	y -= m_rowsOffset;

	assert((x >= 0) && (x < m_numOfCols));
	assert((y >= 0) && (y < m_numOfRows));
	assert((indexPhi >= 0) && (indexPhi < m_pdf4->GetSlicesPerPhi()));
	assert((posLAB >= 0) && (posLAB <= 2));

	return m_pdf6DScale[y][x] * m_pdf4->Get(m_pdf6DSlices[y][x] - 1, indexPhi, indexTheta,
		posAlpha, posBeta, posLAB, tc);
}//--- get -------------------------------------------------------------------

// Computes BTF values for one pixel+angles in RGB space
void
CPDF6D::GetValDeg(int y, int x, float illuminationTheta, float illuminationPhi,
                  const float viewTheta, float viewPhi, float rgb[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;

	// recompute from clockwise to anti-clockwise phi_i notation 
	viewPhi = 360.f - viewPhi;
	while (viewPhi >= 360.f)
		viewPhi -= 360.f;
	while (viewPhi < 0.f)
		viewPhi += 360.f;

	// recompute from clockwise to anti-clockwise phi_v notation: (360.f - phi_i) 
	// rotation of onion-sliced parametrization to be perpendicular to phi_v of viewing angle: - (90.f + phi_v)
	illuminationPhi = (360.f - illuminationPhi) - (90.f + viewPhi);

	while (illuminationPhi >= 360.f)
		illuminationPhi -= 360.f;
	while (illuminationPhi < 0.f)
		illuminationPhi += 360.f;

	// to radians
	illuminationTheta *= (PI / 180.f);
	illuminationPhi *= (PI / 180.f);

	float alpha, beta;
	ConvertThetaPhiToBetaAlpha(illuminationTheta, illuminationPhi, beta, alpha, tc);

	// Back to degrees
	alpha *= (180.f / PI);
	beta *= (180.f / PI);

	m_pdf4->GetVal(m_pdf6DSlices[y][x] - 1, viewPhi, viewTheta, alpha, beta, rgb, tc);

	// we have to multiply it by valid scale factor at the end
	float scale = m_pdf6DScale[y][x];
	for (int i = 0; i < 3; i++)
		rgb[i] *= scale;

	return;
}//--- getValDeg -------------------------------------------------------------

// Computes BTF value for one pixel+angles in RGB space, it avoids
// some interpolation computations as in function getValDeg()
void
CPDF6D::GetValDeg2(int y, int x, float illuminationTheta, float illuminationPhi,
                   const float viewTheta, float viewPhi, float rgb[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;


	// recompute from clockwise to anti-clockwise phi_i notation 
	viewPhi = 360.f - viewPhi;
	while (viewPhi >= 360.f)
		viewPhi -= 360.f;
	while (viewPhi < 0.f)
		viewPhi += 360.f;

	// recompute from clockwise to anti-clockwise phi_v notation: (360.f - phi_i) 
	// rotation of onion-sliced parametrization to be perpendicular to phi_v of viewing angle: - (90.f + phi_v)
	illuminationPhi = (360.f - illuminationPhi) - (90.f + viewPhi);
	while (illuminationPhi >= 360.f)
		illuminationPhi -= 360.f;
	while (illuminationPhi < 0.f)
		illuminationPhi += 360.f;

	// to radians
	illuminationTheta *= (PI / 180.f);
	illuminationPhi *= (PI / 180.f);

	ConvertThetaPhiToBetaAlpha(illuminationTheta, illuminationPhi, tc.m_beta, tc.m_alpha, tc);

	// Back to degrees. Set the values to auxiliary structure
	tc.m_alpha *= (180.f / PI);
	tc.m_beta *= (180.f / PI);

	assert((viewTheta >= 0.f) && (viewTheta <= 90.f));
	assert((viewPhi >= 0.f) && (viewPhi < 360.f));

	// Now we set the object interpolation data
	// For PDF1D and IndexAB, beta coefficient, use correct
	// parameterization
	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	tc.SetForAngleBetaDeg(tc.m_beta);

	// For PDF2D
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	tc.SetForAngleAlphaDeg(tc.m_alpha);
	// For PDF3D
	tc.m_theta = viewTheta;
	tc.m_iTheta = (int)floor(viewTheta / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (viewTheta - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_phi = viewPhi;
	tc.m_iPhi = (int)floor(viewPhi / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi <= tc.m_slicesPerPhi));
	tc.m_wPhi = (viewPhi - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));

	// Now get the value by interpolation between 2 PDF4D, 4 PDF3D,
	// 8 PDF2D, 16 PDF1D, and 16 IndexAB values for precomputed
	// interpolation coefficients and indices
	m_pdf4->GetVal(m_pdf6DSlices[y][x] - 1, rgb, tc);
	// we have to multiply it by valid scale factor at the end
	const float scale = m_pdf6DScale[y][x];
	for (int i = 0; i < 3; i++)
		rgb[i] *= scale;

	return;
}//--- getValDeg2 -------------------------------------------------------------

// ---- here are angles specified in radians[-]
void
CPDF6D::GetValRad(int y, int x, float theta_i, float phi_i,
	float theta_v, float phi_v, float rgb[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;

	// we convert incoming (light) direction to alpha-beta parameterization
	float alpha, beta;

	if (!tc.m_codeBtfFlag) {
		phi_v = 2.0f * PI - phi_v; // This should be done
		while (phi_v > 2.0f * PI)
			phi_v -= 2.0f * PI;
		while (phi_v < 0.f)
			phi_v += 2.0f * PI;

		// we have to correctly specify phi_v as the difference between
		// the angles of outgoing direction and incoming direction
		phi_i = -phi_i - phi_v - PI / 2.0f;

		while (phi_i > 2.0f * PI)
			phi_i -= 2.0f * PI;
		while (phi_i < 0.f)
			phi_i += 2.0f * PI;

		ConvertThetaPhiToBetaAlpha(theta_i, phi_i, beta, alpha, tc);
	}
	else {
		float x = cos(phi_i - phi_v) * sin(theta_i);
		float y = sin(phi_i - phi_v) * sin(theta_i);
		//float z = cos(thetaI);

		beta = asin(Clamp(y, -1.0f, 1.0f));
		float cosBeta = cos(beta);
		if (cosBeta < 0.00001f)
		{
			alpha = 0.0f;
			return;
		}
		float tmp = Clamp(-x / cosBeta, -1.0f, 1.0f);
		alpha = asin(tmp);
	}

	// Convert to degrees for alpha and beta
	alpha *= (180.f / PI);
	beta *= (180.f / PI);
	// Convert to degrees for theta and phi - viewing direction
	theta_v *= (180.f / PI);
	phi_v *= (180.f / PI);

	m_pdf4->GetVal(m_pdf6DSlices[y][x] - 1, phi_v, theta_v, alpha, beta, rgb, tc);
	// we have to multiply it by valid scale factor at the end
	float scale = m_pdf6DScale[y][x];
	for (int i = 0; i < 3; i++)
		rgb[i] *= scale;
	return;
}//--- getValRad ----------------------------------------------------------------

void
CPDF6D::GetValRad2(int y, int x, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;

	if (!tc.m_codeBtfFlag) {
		phi_v = 2.0f * PI - phi_v; // This should be done
		while (phi_v > 2.0f * PI)
			phi_v -= 2.0f * PI;
		while (phi_v < 0.f)
			phi_v += 2.0f * PI;

		// we have to correctly specify phi_v as the difference between
		// the angles of outgoing direction and incoming direction
		phi_i = -phi_i - phi_v - PI / 2.0f;

		while (phi_i > 2.0f * PI)
			phi_i -= 2.0f * PI;
		while (phi_i < 0.f)
			phi_i += 2.0f * PI;

		ConvertThetaPhiToBetaAlpha(theta_i, phi_i, tc.m_beta, tc.m_alpha, tc);
	}
	else {
		float x = cos(phi_i - phi_v) * sin(theta_i);
		float y = sin(phi_i - phi_v) * sin(theta_i);
		//float z = cos(thetaI);

		tc.m_beta = asin(Clamp(y, -1.0f, 1.0f));
		float cosBeta = cos(tc.m_beta);
		if (cosBeta < 0.00001f)
		{
			tc.m_alpha = 0.0f;
			return;
		}
		float tmp = Clamp(-x / cosBeta, -1.0f, 1.0f);
		tc.m_alpha = asin(tmp);
	}

	// Back to degrees. Set the values to auxiliary structure
	tc.m_alpha *= (180.f / PI);
	tc.m_beta *= (180.f / PI);

	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v < 360.f));

	// Now we set the object interpolation data
	// For PDF1D and IndexAB, beta coefficient
	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	tc.SetForAngleBetaDeg(tc.m_beta);

	// For PDF2D
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	tc.SetForAngleAlphaDeg(tc.m_alpha);

	// For viewing direction we go to degrees
	phi_v *= (180.f / PI);
	theta_v *= (180.f / PI);

	// For PDF3D
	tc.m_iTheta = (int)floor(theta_v / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (theta_v - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_iPhi = (int)floor(phi_v / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi <= tc.m_slicesPerPhi - 1));
	tc.m_wPhi = (phi_v - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));
	// Now get the value by interpolation between 2 PDF4D, 4 PDF3D,
	// 8 PDF2D, 16 PDF1D, and 16 IndexAB values for precomputed
	// interpolation coefficients and indices
	m_pdf4->GetVal(m_pdf6DSlices[y][x] - 1, RGB, tc);
	// we have to multiply it by valid scale factor at the end
	float scale = m_pdf6DScale[y][x];
	for (int i = 0; i < 3; i++)
		RGB[i] *= scale;
	return;
}//--- getValRad2 ----------------------------------------------------------------


// Computes BTF value for one pixel+angles in RGB space, it avoids
// some interpolation computations as in function getValDeg()
void
CPDF6D::GetValDegShepard(int y, int x, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;

	// recompute from clockwise to anti-clockwise phi_i notation 
	phi_v = 360.f - phi_v;
	while (phi_v >= 360.f)
		phi_v -= 360.f;
	while (phi_v < 0.f)
		phi_v += 360.f;

	// recompute from clockwise to anti-clockwise phi_v notation: (360.f - phi_i) 
	// rotation of onion-sliced parametrization to be perpendicular to phi_v of viewing angle: - (90.f + phi_v)
	phi_i = (360.f - phi_i) - (90.f + phi_v);
	while (phi_i >= 360.f)
		phi_i -= 360.f;
	while (phi_i < 0.f)
		phi_i += 360.f;

	// to radians
	theta_i *= (PI / 180.f);
	phi_i *= (PI / 180.f);

	ConvertThetaPhiToBetaAlpha(theta_i, phi_i, tc.m_beta, tc.m_alpha, tc);

	// Back to degrees. Set the values to auxiliary structure
	tc.m_alpha *= (180.f / PI);
	tc.m_beta *= (180.f / PI);

	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v < 360.f));

	// Now we set the object interpolation data
	// For PDF1D and IndexAB, beta coefficient, use correct
	// parameterization
	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	tc.SetForAngleBetaDeg(tc.m_beta);

	// For PDF2D
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	tc.SetForAngleAlphaDeg(tc.m_alpha);

	// For PDF3D
	tc.m_theta = theta_v;
	tc.m_iTheta = (int)floor(theta_v / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (theta_v - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_phi = phi_v;
	tc.m_iPhi = (int)floor(phi_v / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi <= tc.m_slicesPerPhi));
	tc.m_wPhi = (phi_v - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));

#ifdef DEBUG_ANGLES
	tc.DebugPrint();
#endif

	// This is the maximum distance we accept for Shepard interpolation
	// all the points in the distance smaller should be included
	tc.m_maxDist = 1.8f;
	tc.m_maxDist2 = Square(tc.m_maxDist);

	// Setting some parameters in tc
	tc.m_wMinBeta2 = Min(1.0f - tc.m_wBeta, tc.m_wBeta);
	tc.m_wMinBeta2 = Square(tc.m_wMinBeta2);

	tc.m_wMinAlpha2 = Min(1.0f - tc.m_wAlpha, tc.m_wAlpha);
	tc.m_wMinAlpha2 = Square(tc.m_wMinAlpha2);

	tc.m_wMinTheta2 = Min(1.0f - tc.m_wTheta, tc.m_wTheta);
	tc.m_wMinTheta2 = Square(tc.m_wMinTheta2);

	// initializing values
	tc.m_rgb[0] = tc.m_rgb[1] = tc.m_rgb[2] = 0.f;
	tc.m_sumWeight = 0.f;
	tc.m_countWeight = 0;

	// Now get the value by interpolation between 2 PDF4D, 4 PDF3D,
	// 8 PDF2D, 16 PDF1D, and 16 IndexAB values for precomputed
	// interpolation coefficients and indices
	m_pdf4->GetValShepard(m_pdf6DSlices[y][x] - 1, tc);

	RGB[0] = tc.m_rgb[0] / tc.m_sumWeight;
	RGB[1] = tc.m_rgb[1] / tc.m_sumWeight;
	RGB[2] = tc.m_rgb[2] / tc.m_sumWeight;

	// we have to multiply it by valid scale factor at the end
	float scale = m_pdf6DScale[y][x];
	for (int i = 0; i < 3; i++)
		RGB[i] *= scale;
	return;
}//--- getValDegShepard -------------------------------------------------------------


// Computes BTF value for one pixel+angles in RGB space, it avoids
// some interpolation computations as in function getValDegSphepard()
void
CPDF6D::GetValDegShepard2(int y, int x, float theta_i, float phi_i,
	float theta_v, float phi_v, float RGB[], TSharedCoordinates& tc) const
{
	x -= m_colsOffset;
	while (x < 0)
		x += m_numOfCols;
	y -= m_rowsOffset;
	while (y < 0)
		y += m_numOfRows;
	x %= m_numOfCols;
	y %= m_numOfRows;

	// recompute from clockwise to anti-clockwise phi_i notation 
	phi_v = 360.f - phi_v;
	while (phi_v >= 360.f)
		phi_v -= 360.f;
	while (phi_v < 0.f)
		phi_v += 360.f;

	// recompute from clockwise to anti-clockwise phi_v notation: (360.f - phi_i) 
	// rotation of onion-sliced parametrization to be perpendicular to phi_v of viewing angle: - (90.f + phi_v)
	phi_i = (360.f - phi_i) - (90.f + phi_v);
	while (phi_i >= 360.f)
		phi_i -= 360.f;
	while (phi_i < 0.f)
		phi_i += 360.f;

	// to radians
	theta_i *= (PI / 180.f);
	phi_i *= (PI / 180.f);

	ConvertThetaPhiToBetaAlpha(theta_i, phi_i, tc.m_beta, tc.m_alpha, tc);

	// Back to degrees. Set the values to auxiliary structure
	tc.m_alpha *= (180.f / PI);
	tc.m_beta *= (180.f / PI);

	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v < 360.f));

	// Now we set the object interpolation data
	// For PDF1D and IndexAB, beta coefficient, use correct
	// parameterization
	assert((tc.m_beta >= -90.f) && (tc.m_beta <= 90.f));
	tc.SetForAngleBetaDeg(tc.m_beta);

	// For PDF2D
	assert((tc.m_alpha >= -90.f) && (tc.m_alpha <= 90.f));
	tc.SetForAngleAlphaDeg(tc.m_alpha);

	// For PDF3D
	tc.m_theta = theta_v;
	tc.m_iTheta = (int)floor(theta_v / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (theta_v - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_phi = phi_v;
	tc.m_iPhi = (int)floor(phi_v / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi < tc.m_slicesPerPhi));
	tc.m_wPhi = (phi_v - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));

#ifdef DEBUG_ANGLES
	tc.DebugPrint();
#endif

	// This is the maximum distance we accept for Shepard interpolation
	// all the points in the distance smaller should be included
	tc.m_maxDist = 1.8f;
	tc.m_maxDist2 = Square(tc.m_maxDist);

	// Setting some parameters in tc
	tc.m_wMinBeta2 = Min(1.0f - tc.m_wBeta, tc.m_wBeta);
	tc.m_wMinBeta2 = Square(tc.m_wMinBeta2);

	tc.m_wMinAlpha2 = Min(1.0f - tc.m_wAlpha, tc.m_wAlpha);
	tc.m_wMinAlpha2 = Square(tc.m_wMinAlpha2);

	tc.m_wMinTheta2 = Min(1.0f - tc.m_wTheta, tc.m_wTheta);
	tc.m_wMinTheta2 = Square(tc.m_wMinTheta2);

	// initializing values
	tc.m_rgb[0] = tc.m_rgb[1] = tc.m_rgb[2] = 0.f;
	tc.m_sumWeight = 0.f;
	tc.m_countWeight = 0;

	int intMaxDist = (int)floorf(tc.m_maxDist);
	float dist2 = 0.f;

#define FASTER_CALL
#ifdef FASTER_CALL
	RGB[0] = RGB[1] = RGB[2] = 0.f;
	tc.m_scale = 1.0f;
#endif

	// PDF4D
	for (int indexPhi = tc.m_iPhi - intMaxDist; indexPhi <= tc.m_iPhi + intMaxDist; indexPhi++) {
		if ((indexPhi < 0) || (indexPhi >= tc.m_slicesPerPhi))
			continue;
		float dist2phi = Square((float)indexPhi - (float)tc.m_iPhi - tc.m_wPhi);
		dist2 = dist2phi;
		// PDF3D
		for (int indexTheta = tc.m_iTheta - intMaxDist; indexTheta <= tc.m_iTheta + intMaxDist; indexTheta++) {
			if ((indexTheta < 0) || (indexTheta >= tc.m_slicesPerTheta))
				continue;
			float dist2theta = Square((float)indexTheta - (float)tc.m_iTheta - tc.m_wTheta);
			dist2 = dist2theta + dist2phi;
			if (dist2 < tc.m_maxDist2) {
				// PDF2D
				for (int indexAlpha = tc.m_iAlpha - intMaxDist; indexAlpha <= tc.m_iAlpha + intMaxDist; indexAlpha++)
				{
					if ((indexAlpha < 0) || (indexAlpha >= tc.m_slicesPerHemi))
						continue;
					float dist2alpha = Square((float)indexAlpha - (float)tc.m_iAlpha - tc.m_wAlpha);
					dist2 = dist2alpha + dist2theta + dist2phi;
					if (dist2 < tc.m_maxDist2) {
						// PDF1D
						for (int indexBeta = tc.m_iBeta - intMaxDist; indexBeta <= tc.m_iBeta + intMaxDist; indexBeta++)
						{
							if ((indexBeta < 0) || (indexBeta >= tc.m_lengthOfSlice))
								continue;
							float dist2beta = Square((float)indexBeta - (float)tc.m_iBeta - tc.m_wBeta);
							dist2 = dist2beta + dist2alpha + dist2theta + dist2phi;
							if (dist2 < tc.m_maxDist2) {
								// We can add this grid point to Shepard interpolation
								// since distance is smaller than maximum prespecified distance
								float weight = Square(tc.m_maxDist - sqrt(dist2)) /
									(dist2 * tc.m_maxDist2);

#ifdef FASTER_CALL
								tc.m_indexBeta = indexBeta;
								tc.m_indexAlpha = indexAlpha;
								tc.m_indexTheta = indexTheta;
								tc.m_indexPhi = indexPhi;
								m_pdf4->GetAll(m_pdf6DSlices[y][x] - 1, tc);

								RGB[0] += tc.m_rgb[0] * weight;
								RGB[1] += tc.m_rgb[1] * weight;
								RGB[2] += tc.m_rgb[2] * weight;
#else // FASTER_CALL
								PDF4->getAll(PDF6Dslices[y][x] - 1, indexPhi,
									indexTheta, indexAlpha, indexBeta, RGB);

								tc.RGB[0] += RGB[0] * weight;
								tc.RGB[1] += RGB[1] * weight;
								tc.RGB[2] += RGB[2] * weight;
#endif // FASTER_CALL
								tc.m_sumWeight += weight;
								tc.m_countWeight++;
							} // if
						} // for beta
					} // if
				} // for alpha
			} // if
		} // for phi
	} // for phi

#ifdef FASTER_CALL
	float scale = m_pdf6DScale[y][x];
	scale /= tc.m_sumWeight;
	RGB[0] *= scale;
	RGB[1] *= scale;
	RGB[2] *= scale;
#else
  // we have to multiply it by valid scale factor at the end
	float scale = PDF6Dscale[y][x];
	scale /= tc.sumWeight;
	RGB[0] = tc.RGB[0] * scale;
	RGB[1] = tc.RGB[1] * scale;
	RGB[2] = tc.RGB[2] * scale;
#endif

	return;
}//--- getValDegShepard2 -------------------------------------------------------------


// get an index of planar position x,y
int
CPDF6D::GetIndex(int y, int x) const
{
	x -= m_colsOffset;
	y -= m_rowsOffset;
	x %= m_numOfCols;
	y %= m_numOfRows;
	assert((x >= 0) && (x < m_numOfCols));
	assert((y >= 0) && (y < m_numOfRows));

	// Here 0 means that the pixel has not been yet computed !
	return m_pdf6DSlices[y][x];
}//--- getIndex -------------------------------------------------------------------

// get scale of planar position x,y
float
CPDF6D::GetScale(int y, int x) const
{
	x -= m_colsOffset;
	y -= m_rowsOffset;
	x %= m_numOfCols;
	y %= m_numOfRows;
	assert((x >= 0) && (x < m_numOfCols));
	assert((y >= 0) && (y < m_numOfRows));

	return m_pdf6DScale[y][x];
}//--- getScale -------------------------------------------------------------------

int
CPDF6D::GetMemory() const
{
	return m_numOfCols * m_numOfRows * (sizeof(int) + sizeof(float));
}//--- getMemory -------------------------------------------------------------

int
CPDF6D::GetMemoryQ() const
{
	// The index to PDF4D
	int bitsForIndex = (int)(ceilf(log2(m_pdf4->GetNoOfPdf4D() + 1)));
	int size = (m_numOfCols * m_numOfRows * bitsForIndex) / 8 + 1; // in Bytes

	// scaling factor
	size += m_numOfCols * m_numOfRows * (sizeof(float));
	return size;
}//--- getMemory -------------------------------------------------------------

int
CPDF6D::Load(char* prefix, int MLF)
{
	assert(prefix);
	assert((MLF == 0) || (MLF == 1));

	CleanObject();

	char fileName[1000];

	// loading data from TXT file
	int nr, nc; float minI, maxI;
	sprintf(fileName, "%s_PDF6Dslices.txt", prefix);
	ReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);
	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	m_numOfCols = nc;
	m_numOfRows = nr;
	Allocate(m_numOfRows, m_numOfCols);

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfRows; irow++)
		for (int jcol = 0; jcol < m_numOfCols; jcol++)
			m_pdf6DSlices[irow][jcol] = tmpiArr[irow][jcol];

	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	sprintf(fileName, "%s_PDF6Dscale.txt", prefix);
	float minF, maxF;
	ReadTxtHeader(fileName, &nr, &nc, &minF, &maxF);
	m_numOfCols = nc;
	m_numOfRows = nr;

	ReadTxt(tmpArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfRows; irow++)
		for (int jcol = 0; jcol < m_numOfCols; jcol++)
			m_pdf6DScale[irow][jcol] = tmpArr[irow][jcol];

	Free2(tmpArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok, loaded
}//--- load ------------------------------------------------------------------

// Computes BTF values for one pixel+angles in RGB space
int
CPDF6D::ImportanceSamplingDeg(int y, int x, float theta_v, float phi_v,
	float q0, float q1, float& theta_i, float& phi_i,
	TSharedCoordinates& tc)
{
	x -= m_colsOffset;
	y -= m_rowsOffset;
	x %= m_numOfCols;
	y %= m_numOfRows;
	assert(x >= 0);
	assert(y >= 0);

	// Checking the random numbers to range [0-1]x[0-1]
	if ((q0 < 0.f) || (q0 > 1.f) ||
		(q1 < 0.f) || (q1 > 1.f)) {
		theta_i = -90.f, phi_i = 0.f; // showing error clearly
		return 1; // error - wrong input value
	}

	phi_v = 360.f - phi_v; // This should be done

	while (phi_v >= 360.f)
		phi_v -= 360.f;
	while (phi_v < 0.f)
		phi_v += 360.f;

	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v < 360.f));

	// For PDF3D
	tc.m_iTheta = (int)floor(theta_v / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (theta_v - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_iPhi = (int)floor(phi_v / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi <= tc.m_slicesPerPhi - 1));
	tc.m_wPhi = (phi_v - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));

	// Now we can make importance sampling
	int res = m_pdf4->ImportanceSamplingDeg(m_pdf6DSlices[y][x] - 1,
		q0, q1, theta_i, phi_i, tc);

	// theta_i and phi_i were computed in degrees

	// We need to correct back this conversion
	phi_i = phi_i + phi_v + 90.f;

	while (phi_i >= 360.f)
		phi_i -= 360.f;
	while (phi_i < 0.f)
		phi_i += 360.f;

	return res; // OK
}//--- importanceSampling -------------------------------------------------------------


// Computes BTF values for one pixel+angles in RGB space
int
CPDF6D::ImportanceSamplingDeg(int iRow, int jCol, const float viewTheta, float viewPhi,
                              const int cntRays, float q0Q1[], float illuminationThetaPhi[],
	TSharedCoordinates& tc) const
{
	jCol -= m_colsOffset;
	iRow -= m_rowsOffset;
	jCol %= m_numOfCols;
	iRow %= m_numOfRows;
	assert(jCol >= 0);
	assert(iRow >= 0);

	viewPhi = 360.f - viewPhi; // This should be done

	while (viewPhi >= 360.f)
		viewPhi -= 360.f;
	while (viewPhi < 0.f)
		viewPhi += 360.f;

	assert((viewTheta >= 0.f) && (viewTheta <= 90.f));
	assert((viewPhi >= 0.f) && (viewPhi <= 360.f));

	// For PDF3D
	tc.m_iTheta = (int)floor(viewTheta / tc.m_stepTheta);
	if (tc.m_iTheta == tc.m_slicesPerTheta)
		tc.m_iTheta = tc.m_slicesPerTheta - 1;
	assert((tc.m_iTheta >= 0) && (tc.m_iTheta <= tc.m_slicesPerTheta - 1));
	tc.m_wTheta = (viewTheta - tc.m_iTheta * tc.m_stepTheta) / tc.m_stepTheta;
	assert((tc.m_wTheta >= -1e-5) && (tc.m_wTheta <= 1.f));
	if (tc.m_wTheta < 0.f) tc.m_wTheta = 0.f;

	// For PDF4D
	tc.m_iPhi = (int)floor(viewPhi / tc.m_stepPhi);
	assert((tc.m_iPhi >= 0) && (tc.m_iPhi <= tc.m_slicesPerPhi - 1));
	tc.m_wPhi = (viewPhi - tc.m_iPhi * tc.m_stepPhi) / tc.m_stepPhi;
	assert((tc.m_wPhi >= 0.f) && (tc.m_wPhi <= 1.f));

	const int res = m_pdf4->ImportanceSamplingDeg(m_pdf6DSlices[iRow][jCol] - 1,
	                                              cntRays, q0Q1, illuminationThetaPhi, tc);

	// theta_i and phi_i were computed in degrees
	for (int i = 0; i < cntRays; i++) {
		// We need to correct back this conversion
		illuminationThetaPhi[2 * i + 1] = illuminationThetaPhi[2 * i + 1] + viewPhi + 90.f;

		while (illuminationThetaPhi[2 * i + 1] >= 360.f)
			illuminationThetaPhi[2 * i + 1] -= 360.f;
		while (illuminationThetaPhi[2 * i + 1] < 0.f)
			illuminationThetaPhi[2 * i + 1] += 360.f;
	} // for all rays

	//  return cntRays; // OK, return number of rays
	return res;
}//--- importanceSampling -------------------------------------------------------------

//! \brief computes albedo for fixed viewer direction
void
CPDF6D::GetViewerAlbedoDeg(int irow, int jcol, float theta_v, float phi_v, float RGB[],
	TSharedCoordinates& tc)
{
	int x = jcol;
	int y = irow;
	x -= m_colsOffset;
	y -= m_rowsOffset;
	x %= m_numOfCols;
	y %= m_numOfRows;
	assert(x >= 0);
	assert(y >= 0);

	phi_v = 360.f - phi_v; // This should be done

	while (phi_v >= 360.f)
		phi_v -= 360.f;
	while (phi_v < 0.f)
		phi_v += 360.f;

	m_pdf4->GetViewerAlbedoDeg(m_pdf6DSlices[y][x] - 1, theta_v, phi_v, RGB, tc);
	float scale = m_pdf6DScale[y][x];
	RGB[0] *= scale;
	RGB[1] *= scale;
	RGB[2] *= scale;
}

