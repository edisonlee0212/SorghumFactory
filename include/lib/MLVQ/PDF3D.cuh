#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF2DSeparate.cuh>
namespace RayTracerFacility
{
	struct PDF3D
	{
		// the used number of 3D functions
		int m_numOfPdf3D;
		// the number of slices per theta (=2D functions) to represent one 3D function
		int m_slicesPerTheta;
		// angle theta quantization step
		float m_stepTheta;
		// the size of the data entry to be used here during restoration
		int m_size3D;

		// These are the data allocated maxPDF2D times, serving to represent the function
		CudaBuffer m_pdf3DSlicesBuffer;
		int* m_pdf3DSlices;
		CudaBuffer m_pdf3DScalesBuffer;
		float* m_pdf3DScales;
		
		// the database of 2D functions to which we point in the array PDF3Dslices
		PDF2DSeparate m_pdf2;

		void Init(const int& slicesPerTheta)
		{
			m_slicesPerTheta = slicesPerTheta;
			m_stepTheta = 90.0f / (slicesPerTheta - 1);
			m_numOfPdf3D = 0;
			m_size3D = m_slicesPerTheta * m_pdf2.m_size2D;
		}
		
		__device__
			void GetVal(const int& pdf3DIndex, glm::vec3& out, SharedCoordinates& tc) const
		{
			const int i = tc.m_iTheta;
			assert(i >= 0 && i < m_slicesPerTheta);
			assert(pdf3DIndex >= 0 && pdf3DIndex < m_numOfPdf3D);
			if (i < m_slicesPerTheta - 1) {
				const float w = tc.m_wTheta;
				glm::vec3 out2;
				m_pdf2.GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i], out, tc);
				m_pdf2.GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i + 1], out2, tc);
				const float s1 = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i] * (1.0f - w);
				const float s2 = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i + 1] * w;
				out = out * s1 + out2 * s2;
			}
			else{
				return;
				m_pdf2.GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i], out, tc);
				const float s = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i];
				out *= s;
			}
		}
	};
}