#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF4D.cuh>
#include <glm/ext/scalar_constants.hpp>
namespace RayTracerFacility
{
	template<typename T>
	struct PDF6D {
		int m_numOfRows;          //! no. of rows in spatial BTF index
		int m_numOfCols;          //! no. of columns in spatial BTF index
		int m_rowsOffset;       //! offset of the first row as we do not need to start from 0
		int m_colsOffset;       //! offset of the first column as we do not need to start from 0  
		int* m_pdf6DSlices;   //! planar index pointing on 4D PDF for individual pixels
		float* m_pdf6DScale; //! corresponding normalization values
		// the database of 4D functions to which we point in the array PDF6Dslices
		PDF4D<T> m_pdf4;
		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		// This is required for SSIM data precomputation
		// the number of slices per phi (=3D functions) to represent one 4D function
		int m_slicesPerPhi;
		// the number of slices per theta (=2D functions) to represent one 3D function
		int m_slicesPerTheta;
		// the number of indices in parameter alpha
		int m_slicesPerHemisphere;
		// the number of values for 1D function
		int m_lengthOfSlice;
		// the number of colors
		int m_numOfColors;
		__device__
			virtual void GetValDeg2(const glm::uvec2& texCoord, float illuminationTheta, float illuminationPhi, float viewTheta, float viewPhi,
			T& out, SharedCoordinates& tc) const
		{
			int x = texCoord.x;
			int y = texCoord.y;
			
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
			illuminationTheta *= glm::pi<float>() / 180.f;
			illuminationPhi *= glm::pi<float>() / 180.f;

			ConvertThetaPhiToBetaAlpha(illuminationTheta, illuminationPhi, tc.m_beta, tc.m_alpha, tc);

			// Back to degrees. Set the values to auxiliary structure
			tc.m_alpha *= 180.f / glm::pi<float>();
			tc.m_beta *= 180.f / glm::pi<float>();

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
			m_pdf4.GetVal(m_pdf6DSlices[y * m_numOfCols + x] - 1, out, tc);
			// we have to multiply it by valid scale factor at the end
			const float scale = m_pdf6DScale[y * m_numOfCols + x];
			out *= scale;
		}
	};
}
