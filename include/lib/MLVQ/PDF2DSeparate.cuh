#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF2D.cuh>
#include <IndexAB.cuh>
#include <PDF1D.cuh>
#include <CIELab.cuh>
namespace RayTracerFacility
{
	struct PDF2DSeparate : PDF2D<glm::vec3> {
		struct PDF2DColor
		{
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
			IndexAB m_iab;

			// Here are the indices to CIndexAB class
			int* m_pdf2DColors;
			__device__
				void GetVal(const int& pdf2DIndex, glm::vec3& out, SharedCoordinates& tc) const
			{
				const int i = tc.m_iAlpha;
				const float w = tc.m_wAlpha;
				glm::vec3 ab1, ab2;
				// colors
				m_iab.GetVal(m_pdf2DColors[pdf2DIndex * m_lengthOfSlice + i], ab1, tc);
				m_iab.GetVal(m_pdf2DColors[pdf2DIndex * m_lengthOfSlice + i + 1], ab2, tc);
				out[1] = ab1[0] * (1.f - w) + ab2[0] * w;
				out[2] = ab1[1] * (1.f - w) + ab2[1] * w;
			}
		};

		struct PDF2DLuminance
		{
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
			PDF1D m_pdf1;

			// Here are the indices to PDF1D class
			int* m_pdf2DSlices;
			// Here are the scale to PDF1D class, PDF1 functions are multiplied by that
			float* m_pdf2DScale;
			// This is optional, not required for rendering, except importance sampling
			float* m_pdf2DNorm;

			__device__
				void GetVal(const int& pdf2DIndex, glm::vec3& out, SharedCoordinates& tc) const
			{
				assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));

				const int i = tc.m_iAlpha;
				const float w = tc.m_wAlpha;

				// This is different to compact representation ! we interpolate in luminances
				const float l1 = m_pdf2DScale[pdf2DIndex * m_lengthOfSlice + i] * m_pdf1.GetVal(m_pdf2DSlices[pdf2DIndex * m_lengthOfSlice + i], tc);
				const float l2 = m_pdf2DScale[pdf2DIndex * m_lengthOfSlice + i + 1] * m_pdf1.GetVal(m_pdf2DSlices[pdf2DIndex * m_lengthOfSlice + i + 1], tc);
				out[0] = (1.f - w) * l1 + w * l2;
			}
		};

		// Here are the instances of color and luminance 2D function database
		PDF2DColor m_color;
		PDF2DLuminance m_luminance;
		// Here are the indices of luminances + color 2D functions
		// index [][0] is luminance, index [][1] is color
		int* m_indexLuminanceColor;
		__device__
		void GetVal(const int& pdf2DIndex, glm::vec3& out, SharedCoordinates& tc) const override
		{
			assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));

			glm::vec3 userCMdata;
			// First, get only luminance
			m_luminance.GetVal(m_indexLuminanceColor[pdf2DIndex * 2 + 0], userCMdata, tc);
			// The get color (A and B)
			m_color.GetVal(m_indexLuminanceColor[pdf2DIndex * 2 + 1], userCMdata, tc);
			// Convert to RGB
			UserCmToRgb(userCMdata, out, tc);
		}
	};
}