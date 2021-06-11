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

		float m_stepAlpha; //! angular step between reflectance values in angle alpha (in degrees)
		bool m_useCosBeta; //! use cos angles

		float m_mPostScale;

		
		int m_lengthOfSlice;  //! Number of measurement points on along slice parametrised by "beta"
		int m_slicesPerHemisphere;  //! Number of slices over hemisphere parametrized by "alpha"
		int m_slicePerTheta;      //! number of different theta viewing angles stored in PDF3D
		int m_slicePerPhi;        //! number of different phi viewing angles stored in PDF4D

		bool m_allMaterialsInOneDatabase; //! if to compress all materials into one database
	//! if view direction represented directly by UBO measurement quantization
		bool m_use34ViewRepresentation;
		bool m_usePdf2CompactRep; //! If we do not separate colors and luminances for 2D functions

		int m_materialCount; //! how many materials are stored in the database

		bool Init(const std::string& materialDirectoryPath);
	};
}