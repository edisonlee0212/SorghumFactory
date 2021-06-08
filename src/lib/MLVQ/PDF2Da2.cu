/* **************************************************************************
\file    PDF2Da2.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  2D PDF database
******************************************************************************/
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <CIELab.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
#include <PDF2Da.hpp>
#include <TBTFbase.hpp>



//! \brief computes importance sampling given the coordinates and [0-1]^2
// random variables. It compute the normalized direction.
int
CPDF2DSeparate::CPDF2DLuminance::ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
	int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
	float q0, float q1, float& illuminationTheta, float& illuminationPhi,
	TSharedCoordinates& tc)
{
	// We basically take four PDF2D functions and find the index that makes the interpolation
	// between the four PDF2D to get the correct 'alpha' angle
	assert(m_slicesPerHemisphere < MaxDeltaCNT);

	// -------------- ALPHA --------------------
	int alpha;
	// Here we have to compute marginal PDF over alpha/beta
	// as the poles are not included in the PDF2Dnorm !
	float delta;
	float delta1, delta2, delta3, delta4;
	float delta1last, delta2last, delta3last, delta4last;
	float delta1sum = 0.f, delta2sum = 0.f, delta3sum = 0.f, delta4sum = 0.f;
	float maxSumAlpha = 0.f;
	for (alpha = 0; alpha < m_slicesPerHemisphere; alpha++) {
		delta1 = w1 * m_pdf2DScale[pdf2DIndex1][alpha];
		delta2 = w2 * m_pdf2DScale[pdf2DIndex2][alpha];
		delta3 = w3 * m_pdf2DScale[pdf2DIndex3][alpha];
		delta4 = w4 * m_pdf2DScale[pdf2DIndex4][alpha];
		m_deltaArray[alpha] = delta1 + delta2 + delta3 + delta4;

		if (alpha > 0) {
			// integral for the index ended by current alpha
			maxSumAlpha += 0.5f * (m_deltaArray[alpha] + m_deltaArray[alpha - 1]);

			delta1sum += 0.5f * (delta1 + delta1last);
			delta2sum += 0.5f * (delta2 + delta2last);
			delta3sum += 0.5f * (delta3 + delta3last);
			delta4sum += 0.5f * (delta4 + delta4last);
		}
		delta1last = delta1;
		delta2last = delta2;
		delta3last = delta3;
		delta4last = delta4;
	} // for alpha

	// In alpha direction we search for this value
	const float searchQ0 = maxSumAlpha * q0;

	// OK, linear search. Remember that PDF1D functions are normalized, so we have
	// to just multiply properly the scales to get the right value
	float sumAlpha = 0.f;
	// The computed values - index + weight for Q0
	int indexQ0 = -1;
	float wQ0 = 0;
	for (alpha = 1; alpha < m_slicesPerHemisphere; alpha++) {
		float sumIntegral = 0.5f * (m_deltaArray[alpha] + m_deltaArray[alpha - 1]);
		if (sumAlpha + sumIntegral >= searchQ0) {
			// The interval started by alpha-1 to alpha
			indexQ0 = alpha - 1;
			// we interpolate between [alpha] and [alpha+1]
			wQ0 = (searchQ0 - sumAlpha) / sumIntegral;
			assert((wQ0 >= 0.0f) && (wQ0 <= 1.0f));
			break;
		}
		sumAlpha += sumIntegral;
		assert(sumAlpha <= maxSumAlpha + 1e-3);
	}
	if (indexQ0 == -1) {
		//    printf("ERROR: bug in importance sampling sumAlpha = %f last delta = %f \n", sumAlpha, delta);
		indexQ0 = m_slicesPerHemisphere - 1;
		wQ0 = 1.0f;
		return 1;
	}

	// compute corresponding alpha
	float alphaV = tc.ComputeAngleAlpha(indexQ0, wQ0);
	// assert( (alphaV >= -90.f)&& (alphaV <= 90.f) );
#ifdef _DEBUG
	if (!((alphaV >= -90.f) && (alphaV <= 90.f))) {
		printf("Problem with alpha conversion - exiting\n");
		printf("indexQ0 = %d wQ0 =%f\n", indexQ0, wQ0);
		alphaV = 0.f;
	}
#endif  

	// -------------- BETA --------------------
	// We have to make normalization again for the second coordinate
	float normC = 1.0f / maxSumAlpha;
	// Let us compute the weights for all PDF4D functions
	w1 = delta1sum * normC;
	w2 = delta2sum * normC;
	w3 = delta3sum * normC;
	w4 = delta4sum * normC;
#ifdef _DEBUG
	float sumw = w1 + w2 + w3 + w4;
	if ((sumw < 0.99f) || (sumw > 1.001f)) {
		printf("Problem with accuracy - sum = %f != 1.0f\n", sumw);
		sumw = 1.0f;
	}
#endif  

	// We have to compute eight weights for 8 1D curves in
	// which we find the value according to q1
	float w11 = w1 * wQ0;
	float w12 = w1 * (1.0f - wQ0);
	float w21 = w2 * wQ0;
	float w22 = w2 * (1.0f - wQ0);
	float w31 = w3 * wQ0;
	float w32 = w3 * (1.0f - wQ0);
	float w41 = w4 * wQ0;
	float w42 = w4 * (1.0f - wQ0);

#ifdef _DEBUG
	float sumw2 = w11 + w12 + w21 + w22 + w31 + w32 + w41 + w42;
	if ((sumw2 < 0.99) || (sumw > 1.01)) {
		printf("IS - problem with accuracy sumw - exiting\n");
		sumw2 = 1.0f;
	}
#endif  

	// The maximum possible value when summing the the value
	float maxSumBeta = 0.f;
	int beta;
	// This is a trivial algorithm that should give always
	// correct results. The number should be the same as above
	for (beta = 0; beta < m_lengthOfSlice; beta++) {
		delta = w11 * m_pdf2DScale[pdf2DIndex1][indexQ0] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex1][indexQ0], beta, tc);
		delta += w12 * m_pdf2DScale[pdf2DIndex1][indexQ0 + 1] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex1][indexQ0 + 1], beta, tc);

		delta += w21 * m_pdf2DScale[pdf2DIndex2][indexQ0] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex2][indexQ0], beta, tc);
		delta += w22 * m_pdf2DScale[pdf2DIndex2][indexQ0 + 1] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex2][indexQ0 + 1], beta, tc);

		delta += w31 * m_pdf2DScale[pdf2DIndex3][indexQ0] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex3][indexQ0], beta, tc);
		delta += w32 * m_pdf2DScale[pdf2DIndex3][indexQ0 + 1] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex3][indexQ0 + 1], beta, tc);

		delta += w41 * m_pdf2DScale[pdf2DIndex4][indexQ0] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex4][indexQ0], beta, tc);
		delta += w42 * m_pdf2DScale[pdf2DIndex4][indexQ0 + 1] *
			m_pdf1->Get(m_pdf2DSlices[pdf2DIndex4][indexQ0 + 1], beta, tc);
		m_deltaArray[beta] = delta;
		if (beta > 0) {
			// compute the integral along curve
			maxSumBeta += (m_deltaArray[beta - 1] + m_deltaArray[beta]) * 0.5f;
		}
		// printf("beta = %d sumBeta = %f\n", beta, maxSumBeta);
	} // for beta

	// The value we search for is this:
	// In alpha direction we search for this value
	float searchQ1 = maxSumBeta * q1;

	// OK, now we can search along coordinate beta, linear way since the
	// discretization is small, binary search will be faster, slightly for N=13;
	int indexQ1 = -1;
	float wQ1 = 0.f;
	float sumBeta = 0.f;
	for (beta = 1; beta < m_lengthOfSlice; beta++) {
		float sumIntegral = 0.5f * (m_deltaArray[beta - 1] + m_deltaArray[beta]);
		if (sumBeta + sumIntegral >= searchQ1) {
			indexQ1 = beta - 1;
			// Just linear interpolation
			wQ1 = (searchQ1 - sumBeta) / sumIntegral;
			assert((wQ1 >= 0.f) && (wQ1 <= 1.0f));
			break;
		}
		sumBeta += sumIntegral;
	} // beta

	if (indexQ1 == -1) {
		// This is perhaps a problem of numerical accuracy, but it should not happen!
		//    printf("ERROR: bug in importance sampling searchQ1 = %3.2f sumBeta = %f" " last delta = %f maxSumBeta = %f\n", searchQ1, sumBeta, delta, maxSumBeta); 
		indexQ1 = m_lengthOfSlice - 1;
		wQ1 = 1.0f;
		return 1;
	}
	// printf("q1=%3.3f indexQ1=%d wQ1 = %3.2f\n", q1, indexQ1, wQ1);

	// ---- here we compute beta - in degrees
	float betaV = tc.ComputeAngleBeta(indexQ1, wQ1);
	assert((betaV >= -90.f) && (betaV <= 90.f));

	// Now convert the direction to radiances
	betaV *= PI / 180.f;
	alphaV *= PI / 180.f;

	// Now we can compute the direction
	ConvertBetaAlphaToThetaPhi(betaV, alphaV, illuminationTheta, illuminationPhi);

#ifdef _DEBUG
	float dir[3];
	ConvertBetaAlphaToXyz(betaV, alphaV, dir);
	if (dir[2] < 0.f) {
		printf("IS - Problem for betaV = %f alphaV = %f dir[2] = %f\n",
			betaV, alphaV, dir[2]);
		abort();
	}
#endif

	// Now convert to degrees
	illuminationTheta *= 180.f / PI;
	illuminationPhi *= 180.f / PI;
	if (illuminationTheta > 90.f) {
		printf("Problem theta = %f\n", illuminationTheta);
	}

	return 0; // OK
}

//! \brief computes importance sampling given the coordinates and [0-1]^2
// random variables. It compute the normalized direction.
int
CPDF2DSeparate::CPDF2DLuminance::ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
	int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
	int cntRays, float q0Q1[], float illuminationThetaPhi[],
	TSharedCoordinates& tc)
{
	// We basically take four PDF2D functions and find the index that makes the interpolation
	// between the four PDF2D to get the correct 'alpha' angle

	int alpha;
	// Here we have to compute marginal PDF over alpha/beta
	// as the poles are not included in the PDF2Dnorm !
	float delta;
	float delta1, delta2, delta3, delta4;
	float delta1last, delta2last, delta3last, delta4last;
	float delta1sum = 0.f, delta2sum = 0.f, delta3sum = 0.f, delta4sum = 0.f;
	float maxSumAlpha = 0.f;
	for (alpha = 0; alpha < m_slicesPerHemisphere; alpha++) {
		delta1 = w1 * m_pdf2DScale[pdf2DIndex1][alpha];
		delta2 = w2 * m_pdf2DScale[pdf2DIndex2][alpha];
		delta3 = w3 * m_pdf2DScale[pdf2DIndex3][alpha];
		delta4 = w4 * m_pdf2DScale[pdf2DIndex4][alpha];
		m_deltaArray[alpha] = delta1 + delta2 + delta3 + delta4;

		if (alpha > 0) {
			// integral for the index ended by current alpha
			maxSumAlpha += 0.5f * (m_deltaArray[alpha] + m_deltaArray[alpha - 1]);

			delta1sum += 0.5f * (delta1 + delta1last);
			delta2sum += 0.5f * (delta2 + delta2last);
			delta3sum += 0.5f * (delta3 + delta3last);
			delta4sum += 0.5f * (delta4 + delta4last);
		}
		delta1last = delta1;
		delta2last = delta2;
		delta3last = delta3;
		delta4last = delta4;
	}

	// We have to make normalization again for the second coordinate
	float normC = 1.0f / maxSumAlpha;
	float w1t = delta1sum * normC;
	float w2t = delta2sum * normC;
	float w3t = delta3sum * normC;
	float w4t = delta4sum * normC;
#ifdef _DEBUG
	float sumw = w1t + w2t + w3t + w4t;
	if ((sumw < 0.99f) || (sumw > 1.001f)) {
		printf("1 Sum = %f != 1.0f\n", sumw);
	}
#endif

	// an auxiliary array to keep the values computed for further search
	float* arrayBeta = new float[m_lengthOfSlice];
	assert(arrayBeta);
	// We just finished computation of albedo - integral over the 
	// hemisphere given the viewer direction

	// Now for all rays
	for (int i = 0; i < cntRays; i++) {
		// In alpha direction we search for this value
		const float searchQ0 = maxSumAlpha * q0Q1[2 * i];

		// OK, linear search. Remember that PDF1D functions are normalized, so we have
		// to just multiply properly the scales to get the right value
		float sumAlpha = 0.f;
		// The computed values - index + weight for Q0
		int indexQ0 = -1;
		float wQ0 = 0.f;
		for (alpha = 1; alpha < m_slicesPerHemisphere; alpha++) {
			float sumIntegral = 0.5f * (m_deltaArray[alpha] + m_deltaArray[alpha - 1]);
			if (sumAlpha + sumIntegral >= searchQ0) {
				indexQ0 = alpha - 1;
				// we interpolate between [alpha] and [alpha+1]
				wQ0 = (searchQ0 - sumAlpha) / sumIntegral;
				assert((wQ0 >= 0.0f) && (wQ0 <= 1.0f));
				break;
			}
			sumAlpha += sumIntegral;
		} // for alpha

		if (indexQ0 == -1) { // Some inaccuracy - this should not happen ....
		  //      printf("ERROR: bug in importance sampling sumAlpha = %f last delta = %f\n", sumAlpha, delta);
			indexQ0 = m_slicesPerHemisphere - 1;
			wQ0 = 1.0f;
			return 1;
		}

		// compute corresponding alpha
		float alphaV = tc.ComputeAngleAlpha(indexQ0, wQ0);
#ifdef _DEBUG
		if (!((alphaV >= -90.f) && (alphaV <= 90.f))) {
			printf("Problem with alpha conversion - exiting\n");
			printf("indexQ0 = %d wQ0 =%f\n", indexQ0, wQ0);
			alphaV = 0.f;
		}
#endif  

		// We have to compute eight weights for 8 1D curves in
		// which we find the value according to q1
		float w11 = w1t * wQ0;
		float w12 = w1t * (1.0f - wQ0);
		float w21 = w2t * wQ0;
		float w22 = w2t * (1.0f - wQ0);
		float w31 = w3t * wQ0;
		float w32 = w3t * (1.0f - wQ0);
		float w41 = w4t * wQ0;
		float w42 = w4t * (1.0f - wQ0);

#ifdef _DEBUG
		float sumw2 = w11 + w12 + w21 + w22 + w31 + w32 + w41 + w42;
		if ((sumw2 < 0.99) || (sumw > 1.01)) {
			printf("IS - problem with sumw - exiting - accuracy sumW2 =%f\n", sumw2);
		}
#endif  

		// The maximum possible value when summing the the value
		float maxSumBeta = 0.f;
		int beta;
		// This is a trivial algorithm that should give always
		// correct results. The number should be the same as above
		for (beta = 0; beta < m_lengthOfSlice; beta++) {
			delta = w11 * m_pdf2DScale[pdf2DIndex1][indexQ0] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex1][indexQ0], beta, tc);
			delta += w12 * m_pdf2DScale[pdf2DIndex1][indexQ0 + 1] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex1][indexQ0 + 1], beta, tc);

			delta += w21 * m_pdf2DScale[pdf2DIndex2][indexQ0] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex2][indexQ0], beta, tc);
			delta += w22 * m_pdf2DScale[pdf2DIndex2][indexQ0 + 1] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex2][indexQ0 + 1], beta, tc);

			delta += w31 * m_pdf2DScale[pdf2DIndex3][indexQ0] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex3][indexQ0], beta, tc);
			delta += w32 * m_pdf2DScale[pdf2DIndex3][indexQ0 + 1] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex3][indexQ0 + 1], beta, tc);

			delta += w41 * m_pdf2DScale[pdf2DIndex4][indexQ0] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex4][indexQ0], beta, tc);
			delta += w42 * m_pdf2DScale[pdf2DIndex4][indexQ0 + 1] *
				m_pdf1->Get(m_pdf2DSlices[pdf2DIndex4][indexQ0 + 1], beta, tc);
			arrayBeta[beta] = delta;
			if (beta > 0) {
				// compute the integral along curve
				maxSumBeta += (arrayBeta[beta - 1] + arrayBeta[beta]) * 0.5f;
			}
			// printf("beta = %d sumBeta = %f\n", beta, maxSumBeta);
		} // for beta

		// The value we search for is this:
		// In alpha direction we search for this value
		float searchQ1 = maxSumBeta * q0Q1[2 * i + 1];

		// OK, now we can search along coordinate beta, linear way since the
		// discretization is small, binary search will be faster, slightly for N=13;
		int indexQ1 = -1;
		float wQ1 = 0.f;
		float sumBeta = 0.f;
		for (beta = 1; beta < m_lengthOfSlice; beta++) {
			float sumIntegral = 0.5f * (arrayBeta[beta - 1] + arrayBeta[beta]);
			if (sumBeta + sumIntegral >= searchQ1) {
				indexQ1 = beta - 1;
				// Just linear interpolation
				wQ1 = (searchQ1 - sumBeta) / sumIntegral;
				assert((wQ1 >= 0.f) && (wQ1 <= 1.0f));
				break;
			}
			sumBeta += sumIntegral;
		} // for beta

		if (indexQ1 == -1) {
			//      printf("ERROR: bug in importance sampling searchQ1 = %3.2f sumBeta = %f" " last delta = %f maxSumBeta = %f\n", searchQ1, sumBeta, delta, maxSumBeta); 
			// Numerical inaccuracy
			indexQ1 = m_lengthOfSlice - 1;
			wQ1 = 1.0f;
			return 1;
		}
		// printf("q1=%3.3f indexQ1=%d wQ1 = %3.2f\n", q1, indexQ1, wQ1);

		// ---- here we compute beta - in degrees
		float betaV = tc.ComputeAngleBeta(indexQ1, wQ1);
		assert((betaV >= -90.f) && (betaV <= 90.f));

		// Now convert the direction to radiances
		betaV *= PI / 180.f;
		alphaV *= PI / 180.f;
		float theta_i, phi_i;
		// Now we can compute the direction
		ConvertBetaAlphaToThetaPhi(betaV, alphaV, theta_i, phi_i);

#ifdef _DEBUG
		float dir[3];
		ConvertBetaAlphaToXyz(betaV, alphaV, dir);
		if (dir[2] < 0.f) {
			printf("IS - Problem for betaV = %f alphaV = %f dir[2] = %f\n",
				betaV, alphaV, dir[2]);
			abort();
		}
#endif

		// Now convert to degrees
		theta_i *= 180.f / PI;
		phi_i *= 180.f / PI;
		if (theta_i > 90.f) {
			printf("Problem theta = %f\n", theta_i);
		}
		// save it to the output array
		illuminationThetaPhi[2 * i] = theta_i;
		illuminationThetaPhi[2 * i + 1] = phi_i;
	} // for all rays

	delete[]arrayBeta;

	return 0; // OK
}

// compute the albedo for fixed viewer direction
void
CPDF2DSeparate::CPDF2DLuminance::GetViewerAlbedoDeg(int pdf2DIndex, float RGB[],
	float& normPar, TSharedCoordinates& tc)
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	float** restoredValuesOpt;
	restoredValuesOpt = Allocation2(0, 2, 0, m_size2D - 1);
	assert(restoredValuesOpt);

	// We compute the luminance of 2D slice from luminance
	this->RestoreLum(restoredValuesOpt, pdf2DIndex, tc);

	float sumRGB[3];
	sumRGB[0] = sumRGB[1] = sumRGB[2] = 0.f;
	float sumL = 0.f;
	// now compute the average of the values
	for (int i = 0; i < m_size2D; i++) {
		// convert to RGB
		float UserCMdata[3], RGB[3];
		// This is luminance
		UserCMdata[0] = restoredValuesOpt[0][i];
		sumL += UserCMdata[0];
		UserCMdata[1] = restoredValuesOpt[1][i];
		UserCMdata[2] = restoredValuesOpt[2][i];
		UserCmToRgb(UserCMdata, RGB, tc);
		sumRGB[0] += RGB[0]; sumRGB[1] += RGB[1]; sumRGB[2] += RGB[2];
	} // for i
	float mult = 1.0f / (float)m_size2D;
	sumRGB[0] *= mult;
	sumRGB[1] *= mult;
	sumRGB[2] *= mult;

#if 0
#ifdef _DEBUG
	if (fabs(PDF2Dnorm[PDF2Dindex] - sumL) / sumL > 0.05f) {
		printf("Normpar = %3.2f does not fit to albedo =%3.2f\n",
			PDF2Dnorm[PDF2Dindex], sumL);
	}
#endif  
#endif

	// copy the result to the output
	normPar = sumL * mult;
	RGB[0] = sumRGB[0];
	RGB[1] = sumRGB[1];
	RGB[2] = sumRGB[2];

	return;
}

// ====================================================================================
// random variables. It compute the normalized direction.
int
CPDF2DSeparate::ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
	int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
	float q0, float q1, float& illuminationTheta, float& illuminationPhi,
	TSharedCoordinates& tc)
{
	// Pass the call with making proper dereferencing to the indices !
	return m_luminance->ImportanceSampling(m_indexLuminanceColor[pdf2DIndex1][0], w1,
		m_indexLuminanceColor[pdf2DIndex2][0], w2,
		m_indexLuminanceColor[pdf2DIndex3][0], w3,
		m_indexLuminanceColor[pdf2DIndex4][0], w4,
		q0, q1, illuminationTheta, illuminationPhi, tc);
}

// random variables. It compute the normalized direction.
int
CPDF2DSeparate::ImportanceSampling(int pdf2DIndex1, float w1, int pdf2DIndex2, float w2,
	int pdf2DIndex3, float w3, int pdf2DIndex4, float w4,
	int cntRays, float q0Q1[], float illuminationThetaPhi[],
	TSharedCoordinates& tc)
{
	// Pass the call with making proper dereferencing to the indices !
	return m_luminance->ImportanceSampling(m_indexLuminanceColor[pdf2DIndex1][0], w1,
		m_indexLuminanceColor[pdf2DIndex2][0], w2,
		m_indexLuminanceColor[pdf2DIndex3][0], w3,
		m_indexLuminanceColor[pdf2DIndex4][0], w4,
		cntRays, q0Q1, illuminationThetaPhi, tc);
}


// =====================================================================================
// =====================================================================================
// =====================================================================================
// =====================================================================================

// constructor, allocates maximum number of 2D PDFs, the number of slices per 2D PDF,
// 1D PDFs database, 1D colour index slices database, metric used for comparison
CPDF2DSeparate::CPDF2DSeparate(int maxPdf2D, int slicesPerHemisphere, CPDF1D* pdf1, CIndexAB* iab,
	int metric, bool belowThresholdSearch, bool insertNewDataIntoCache) :
	CPDF2D(maxPdf2D, slicesPerHemisphere, pdf1, iab)
{
	int LengthOfSlice = pdf1->GetSliceLength();
	this->m_size2D = slicesPerHemisphere * LengthOfSlice;

	m_color = new CPDF2DColor(maxPdf2D, LengthOfSlice, slicesPerHemisphere, iab, 0);
	assert(m_color);

	m_luminance = new CPDF2DLuminance(maxPdf2D, slicesPerHemisphere, pdf1, metric, 0);
	assert(m_luminance);

	// This is the array of indices to separate luminance and color representations
	m_indexLuminanceColor = IAllocation2(0, maxPdf2D - 1, 0, 1);
	assert(m_indexLuminanceColor);
} // ----------------------- CPDF2Dseparate() --------------------------------


CPDF2DSeparate::~CPDF2DSeparate()
{
	delete m_color;
	m_color = 0;
	delete m_luminance;
	m_luminance = 0;

	if (m_indexLuminanceColor)
		IFree2(m_indexLuminanceColor, 0, m_maxPdf2D - 1, 0, 1);
	m_indexLuminanceColor = 0;

} // ----------------------- ~CPDF2Dseparate() --------------------------------

void
CPDF2DSeparate::DeleteData()
{
	if (m_indexLuminanceColor)
		IFree2(m_indexLuminanceColor, 0, m_maxPdf2D - 1, 0, 1);

	m_maxPdf2D = 1;
	m_numOfPdf2D = 0;

	// This is the array of indices to separate luminance and color representations
	m_indexLuminanceColor = IAllocation2(0, m_maxPdf2D - 1, 0, 1);
	assert(m_indexLuminanceColor);
} // ----------------------- ~CPDF2Dseparate() --------------------------------
void CPDF2DSeparate::DeleteDataLuminance() const
{
	assert(m_luminance);
	m_luminance->DeleteData();
}

void CPDF2DSeparate::DeleteDataColor() const
{
	assert(m_color);
	m_color->DeleteData();
}


// check & reallocation of database if needed
void
CPDF2DSeparate::Reallocate()
{
	// reallocation of individual databases if needed
	if (m_numOfPdf2D >= m_maxPdf2D) {
		int newMaxPDF2D = Max(m_maxPdf2D + 5000, m_numOfPdf2D);
		m_indexLuminanceColor = IReallocation2Rows(m_indexLuminanceColor, 0, m_maxPdf2D - 1, newMaxPDF2D - 1,
			0, 1);
		m_maxPdf2D = newMaxPDF2D;
	}

	return;
} // ----------------------- realloc() --------------------------------

// get single value for arbitrary angles alpha,beta and slice index
void
CPDF2DSeparate::GetVal(int pdf2DIndex, float alpha, float beta, float rgb[], TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((beta >= -90.f) && (beta <= 90.f));
	assert((alpha >= -90.f) && (alpha <= 90.f));

	float UserCMdata[3];
	// First, get only luminance
	m_luminance->GetVal(m_indexLuminanceColor[pdf2DIndex][0], alpha, beta, UserCMdata, tc);
	// The get color (A and B)
	m_color->GetVal(m_indexLuminanceColor[pdf2DIndex][1], alpha, beta, UserCMdata, tc);
	// Convert to RGB
	UserCmToRgb(UserCMdata, rgb, tc);

	return;
}  // ---------------------- getVal() ----------------------------------------

// Here alpha and beta is specified by 'tc'
void
CPDF2DSeparate::GetVal(int PDF2Dindex, float rgb[], TSharedCoordinates& tc) const
{
	assert((PDF2Dindex >= 0) && (PDF2Dindex < m_numOfPdf2D));

	float UserCMdata[3];
	// First, get only luminance
	m_luminance->GetVal(m_indexLuminanceColor[PDF2Dindex][0], UserCMdata, tc);
	// The get color (A and B)
	m_color->GetVal(m_indexLuminanceColor[PDF2Dindex][1], UserCMdata, tc);
	// Convert to RGB
	UserCmToRgb(UserCMdata, rgb, tc);
}  // ---------------------- getVal() ----------------------------------------


// Here alpha and beta is specified by 'tc'
void
CPDF2DSeparate::GetValShepard(int PDF2Dindex, float scale, float sumDist2, TSharedCoordinates& tc) const
{
	assert((PDF2Dindex >= 0) && (PDF2Dindex < m_numOfPdf2D));

	float UserCMdata[3];

	float pd2 = sumDist2 + tc.m_wMinBeta2;
	int iMaxDist = (int)(sqrt(tc.m_maxDist2 - sumDist2));
	const float w = tc.m_wAlpha;
	const int i = tc.m_iAlpha;
	for (int ii = i - iMaxDist; (ii <= i); ii++) {
		if (ii >= 0) {
			float minDist2Bound = Square(w + (float)ii - (float)i);
			if (minDist2Bound + pd2 < tc.m_maxDist2) {
				// there is a chance of having grid point value in the distance smaller than specified
				bool a = m_color->GetValShepard(m_indexLuminanceColor[PDF2Dindex][1], ii, sumDist2, UserCMdata, tc);
				if (a) {
					m_luminance->GetValShepard(m_indexLuminanceColor[PDF2Dindex][0], ii, scale, minDist2Bound + sumDist2, UserCMdata, tc);
				}
			}
		}
	}
	for (int ii = i + 1; (ii < m_numOfSlicesPerHemisphere) && (ii <= i + 1 + iMaxDist); ii++) {
		// The distance along 
		float minDist2Bound = Square((float)ii - (float)i - w);
		if (minDist2Bound + pd2 < tc.m_maxDist2) {
			// there is a chance of having grid point value in the distance smaller than specified
			bool a = m_color->GetValShepard(m_indexLuminanceColor[PDF2Dindex][1], ii, sumDist2, UserCMdata, tc);
			if (a) {
				m_luminance->GetValShepard(m_indexLuminanceColor[PDF2Dindex][0], ii, scale, minDist2Bound + sumDist2, UserCMdata, tc);
			}
		}
	}
}  // ---------------------- getValShepard() ----------------------------------------


//return memory in bytes required by the data representation
int
CPDF2DSeparate::GetMemory() const
{
	// just indices
	return m_numOfPdf2D * (2 * sizeof(int));
} // ---------------------- getMemory() ----------------------------------------

int
CPDF2DSeparate::GetMemoryQ() const
{
	// max size of indix to colours
	int bitsForIndex = (int)(ceilf(log2(m_color->GetNumOfPdf2DColor() + 1)));
	// adding max size index to luminances
	bitsForIndex += (int)(ceilf(log2(m_luminance->GetNumOfPdf2DLuminance() + 1)));

	// total size of indices
	return (m_numOfPdf2D * bitsForIndex) / 8 + 1;
} // --------------------- CPDF2Dcolor::getMemoryQ ----------------------------

int
CPDF2DSeparate::Load(char* prefix, int mlf, int maxPdf1D,
	int maxIndexColor, int algPut)
{
	assert(m_color);
	int readStatus = m_color->Load(prefix, mlf, maxIndexColor, algPut);
	assert(m_luminance);
	readStatus |= m_luminance->Load(prefix, mlf, maxPdf1D, algPut);
	if (readStatus)
		return 1; // problem

	assert(prefix);
	char fileName[1000];

	// loading data from TXT files
	int nr, nc, minI, maxI;
	sprintf(fileName, "%s_PDF2Dindices.txt", prefix);
	IReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);

	assert(nc == 2);
	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	m_numOfPdf2D = nr;
	Reallocate();

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	//const int channels = 3;
	for (int irow = 0; irow < m_numOfPdf2D; irow++) {
		for (int jcol = 0; jcol < 2; jcol++)
			m_indexLuminanceColor[irow][jcol] = tmpiArr[irow][jcol];
	} // ----------------------------

	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok
} // ---------------------- load() ----------------------------------------
int CPDF2DSeparate::GetNumOfPdf2DLuminance() const
{
	return m_luminance->GetNumOfPdf2DLuminance();
}

int CPDF2DSeparate::GetMemoryLuminance() const
{
	return m_luminance->GetMemory();
}

int CPDF2DSeparate::GetMemoryLuminanceQ() const
{
	return m_luminance->GetMemoryQ();
}

int CPDF2DSeparate::GetNumOfPdf2DColor() const
{
	return m_color->GetNumOfPdf2DColor();
}

int CPDF2DSeparate::GetMemoryColor() const
{
	return m_color->GetMemory();
}

int CPDF2DSeparate::GetMemoryColorQ() const
{
	return m_color->GetMemoryQ();
}


// compute the albedo for fixed viewer direction
void
CPDF2DSeparate::GetViewerAlbedoDeg(int pdf2DIndex, float rgb[], float& normPar, TSharedCoordinates& tc)
{
	assert(m_luminance);

	return  m_luminance->GetViewerAlbedoDeg(m_indexLuminanceColor[pdf2DIndex][0], rgb, normPar, tc);
}

