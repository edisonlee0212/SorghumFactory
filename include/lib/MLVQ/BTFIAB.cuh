#pragma once
#include <Optix7.hpp>
#include <BTFBase.cuh>
namespace RayTracerFacility
{
	struct BTFIAB : BtfBase<glm::vec3>
	{
		int m_materialOrder; //! order of the material processed
		int m_nColor;              //! number of spectral channels in BTF data

		int m_maxPdf1D;       //! number of allocated 1D PDF L-slices
		int m_maxVectorColor; //! number of allocated CIE a-b colours
		int m_maxIndexSlices; //! number of allocated 1D colour index slices
		int m_maxPdf2D;       //! number of allocated 2D PDF indices
		int m_maxPdf2DLuminanceColor; //! number of allocated 2D PDF indices for color and luminance
		int m_maxPdf3D;       //! number of allocated 3D PDF indices
		int m_maxPdf4D;       //! number of allocated 4D PDF indices
		int m_maxPdf34D;       //! number of allocated 3-4D PDF indices for PDF34D
		bool m_useCosBeta; //! use cos angles

		float m_mPostScale;
		bool Init(const std::string& parName);
		bool LoadBtfBase(const char* prefix, bool recover);
		
	};
}