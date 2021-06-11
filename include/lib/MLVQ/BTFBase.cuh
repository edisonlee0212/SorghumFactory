#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF6D.cuh>
#include <glm/glm.hpp>
namespace RayTracerFacility
{
	struct BtfBase
	{
		SharedCoordinates m_tcTemplate;
		PDF6D m_pdf6;
		bool m_hdr = false;
		float m_hdrValue = 1.0f;
		__device__
		void GetValueDeg(const glm::vec2& texCoord, const float& illuminationTheta, const float& illuminationPhi,
		                 const float& viewTheta, const float& viewPhi, glm::vec3& out, const bool& print) const
		{
			if (illuminationTheta > 90.f || viewTheta > 90.f) {
				out = glm::vec3(1.0f);
				return;
			}
			SharedCoordinates tc(m_tcTemplate);
			// fast version, pre-computation of interpolation values only once
			if (print) printf("Sampling from PDF6...");
			m_pdf6.GetValDeg2(texCoord, illuminationTheta, illuminationPhi, viewTheta, viewPhi, out, tc, print);
			if (print) printf("Col6[%.2f, %.2f, %.2f]\n", out.x, out.y, out.z);
			/*
			if (m_hdr) {
				// we encode the values multiplied by a user coefficient
				// before it is converted to User Color Model
				// Now we have to multiply it back.    
				const float multi = 1.0f / m_hdrValue;
				out *= multi;
			}*/
		}
		
		int m_materialOrder; //! order of the material processed
		int m_nColor;              //! number of spectral channels in BTF data

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
