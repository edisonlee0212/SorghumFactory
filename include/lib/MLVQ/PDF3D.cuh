#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF2D.cuh>
namespace RayTracerFacility
{
	template<typename T>
	struct PDF3D
	{
		// the number of allocated 3D functions to be stored
		int m_maxPdf3D;
		// the used number of 3D functions
		int m_numOfPdf3D;
		// the number of slices per theta (=2D functions) to represent one 3D function
		int m_slicesPerTheta;
		// angle theta quantization step
		float m_stepTheta;
		// the size of the data entry to be used here during restoration
		int m_size3D;

		// These are the data allocated maxPDF2D times, serving to represent the function
		int* m_pdf3DSlices;
		float* m_pdf3DScales;
		
		// the database of 2D functions to which we point in the array PDF3Dslices
		PDF2D<T>* m_pdf2;
		__device__
		void GetVal(const int& pdf3DIndex, T& out, SharedCoordinates& tc) const
		{
			assert((pdf3DIndex >= 0) && (pdf3DIndex < m_numOfPdf3D));
			const int i = tc.m_iTheta;
			if (i < m_slicesPerTheta - 1) {
				const float w = tc.m_wTheta;
				// interpolation between two values retrieved from PDF2D
				T out2;
				m_pdf2->GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i], out, tc);
				m_pdf2->GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i + 1], out2, tc);
				const float s1 = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i] * (1.f - w);
				const float s2 = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i + 1] * w;
				out = out * s1 + out2 * s2;
			}
			else {
				m_pdf2->GetVal(m_pdf3DSlices[pdf3DIndex * m_slicesPerTheta + i], out, tc);
				const float s = m_pdf3DScales[pdf3DIndex * m_slicesPerTheta + i];
				out *= s;
			}
		}
	};
}