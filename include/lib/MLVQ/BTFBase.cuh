#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <PDF6D.cuh>
#include <glm/glm.hpp>
namespace RayTracerFacility
{
	template <typename T>
	struct BtfBase
	{
		SharedCoordinates m_tcTemplate;
		PDF6D<T> m_pdf6;
		bool m_hdr = false;
		float m_hdrValue = 1.0f;
		__device__
		virtual void GetValueDeg(const glm::uvec2& texCoord, const float& illuminationTheta, const float& illuminationPhi,
		                 const float& viewTheta, const float& viewPhi, T& out) const
		{
			if (illuminationTheta > 90.f || viewTheta > 90.f) {
				out = T();
				return;
			}
			SharedCoordinates tc(m_tcTemplate);
			// fast version, pre-computation of interpolation values only once
			m_pdf6.GetValDeg2(texCoord, illuminationTheta, illuminationPhi, viewTheta, viewPhi, out, tc);

			if (m_hdr) {
				// we encode the values multiplied by a user coefficient
				// before it is converted to User Color Model
				// Now we have to multiply it back.    
				const float multi = 1.0f / m_hdrValue;
				out *= multi;
			}
		}
		
		
	};
}
