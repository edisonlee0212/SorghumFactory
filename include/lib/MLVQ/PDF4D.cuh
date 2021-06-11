#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF3D.cuh>
#include <glm/glm.hpp>
namespace RayTracerFacility
{
	struct PDF4D {
		// the used number of 4D functions
		int m_numOfPdf4D;
		// the number of slices per phi (=3D functions) to represent one 4D function
		int m_slicesPerPhi;
		// angle phi quantization step
		float m_stepPhi;
		// the size of the data entry to be used here during restoration
		int m_size4D;

		// These are the data allocated maxPDF4D times, serving to represent the function
		CudaBuffer m_pdf4DSlicesBuffer;
		int* m_pdf4DSlices;
		CudaBuffer m_pdf4DScalesBuffer;
		float* m_pdf4DScales;
		
		PDF3D m_pdf3;

		void Init(const int& slicePerPhi)
		{
			m_slicesPerPhi = slicePerPhi;
			m_stepPhi = 360.0f / slicePerPhi;
			m_numOfPdf4D = 0;
			m_size4D = m_pdf3.m_size3D * slicePerPhi;
		}
		__device__
			void GetVal(const int& pdf4DIndex, glm::vec3& out, SharedCoordinates& tc) const
		{
			const int i = tc.m_iPhi;
			const float w = tc.m_wPhi;
			assert(i >= 0 && i < m_slicesPerPhi);
			assert(pdf4DIndex >= 0 && pdf4DIndex < m_numOfPdf4D);
			if (i < m_slicesPerPhi - 1) {
				glm::vec3 out2;
				m_pdf3.GetVal(m_pdf4DSlices[pdf4DIndex * m_slicesPerPhi + i], out, tc);
				m_pdf3.GetVal(m_pdf4DSlices[pdf4DIndex * m_slicesPerPhi + i + 1], out2, tc);
				const float s1 = m_pdf4DScales[pdf4DIndex * m_slicesPerPhi + i] * (1.0f - w);
				const float s2 = m_pdf4DScales[pdf4DIndex * m_slicesPerPhi + i + 1] * w;
				out = out * s1 + out2 * s2;
			}else
			{
				return;
				m_pdf3.GetVal(m_pdf4DSlices[pdf4DIndex * m_slicesPerPhi + i], out, tc);
			}
		}
	};
}
