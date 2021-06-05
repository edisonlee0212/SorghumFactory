#pragma once
#include <Optix7.hpp>
namespace RayMLVQ
{
	struct Mesh
	{
		glm::vec3* m_position;
		glm::uvec3* m_triangle;
		glm::vec3* m_normal;
		glm::vec3* m_tangent;
		glm::vec4* m_color;
		glm::vec2* m_texCoord;
	};
	
	struct DefaultMaterial {
		glm::vec3  m_surfaceColor;
		float m_roughness = 15;
		float m_metallic = 0.5;
		cudaTextureObject_t m_albedoTexture;
		cudaTextureObject_t m_normalTexture;
		float m_diffuseIntensity;
	};

	struct RayMLVQMaterial {
		
	};

	struct DefaultSbtData
	{
		Mesh m_mesh;
		DefaultMaterial m_material;
	};

	struct RayMLVQSbtData
	{
		Mesh m_mesh;
		bool m_enableMLVQ;
		DefaultMaterial m_material;
		RayMLVQMaterial m_rayMlvqMaterial;
	};
	
	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		DefaultSbtData m_data;
	};

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultIlluminationEstimationRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultIlluminationEstimationRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultIlluminationEstimationRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		DefaultSbtData m_data;
	};

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayMLVQRenderingRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayMLVQRenderingRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayMLVQRenderingRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		RayMLVQSbtData m_data;
	};
}
