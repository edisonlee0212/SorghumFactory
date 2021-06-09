#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF3D.cuh>
#include <glm/glm.hpp>
namespace RayTracerFacility
{
	template<typename T>
	struct CPDF4D {
		// the number of allocated 4D functions to be stored
		int m_maxPdf4D;
		// the used number of 4D functions
		int m_numOfPdf4D;
		// the number of slices per phi (=3D functions) to represent one 4D function
		int m_slicesPerPhi;
		// angle phi quantization step
		float m_stepPhi;
		// the size of the data entry to be used here during restoration
		int m_size4D;

		// These are the data allocated maxPDF4D times, serving to represent the function
		int* m_pdf4DSlices;
		float* m_pdf4DScale;
		
		CPDF3D<T> m_pdf3;

		__device__
		void GetVal(const int& pdf4DIndex, T& out, TSharedCoordinates& tc) const
		{
			const int i = tc.m_iPhi;
			int i2 = i + 1;
			if (i2 == m_slicesPerPhi)
				i2 = 0;
			const float w = tc.m_wPhi;

			// interpolation between two values retrieved from PDF2D
			T out2;
			m_pdf3.GetVal(m_pdf4DSlices[pdf4DIndex * m_slicesPerPhi + i], out, tc);
			m_pdf3.GetVal(m_pdf4DSlices[pdf4DIndex * m_slicesPerPhi + i2], out2, tc);
			const float s1 = m_pdf4DScale[pdf4DIndex * m_slicesPerPhi + i] * (1 - w);
			const float s2 = m_pdf4DScale[pdf4DIndex * m_slicesPerPhi + i2] * w;
			out = out * s1 + out2 * s2;
		}
	};
}
