#pragma once
#include <CUDAModule.hpp>
#include <glm/glm.hpp>

namespace RayMLVQ {
	enum class DefaultRenderingRayType
	{
		RadianceRayType,
		ShadowRayType,
		RayTypeCount
	};
	enum class DefaultIlluminationEstimationRayType
	{
		RadianceRayType,
		RayTypeCount
	};

	enum class RayMLVQRenderingRayType
	{
		RadianceRayType,
		RayTypeCount
	};
	
	struct VertexInfo;
	

	struct DefaultRenderingLaunchParams
	{
		DefaultRenderingProperties m_defaultRenderingProperties;
		struct {
			cudaSurfaceObject_t m_outputTexture;
			size_t m_frameId;
		} m_frame;
		struct {
			cudaTextureObject_t m_environmentalMaps[6];
			float m_lightSize = 1.0f;
			glm::vec3 m_direction = glm::vec3(0, -1, 0);
		} m_skylight;
		OptixTraversableHandle m_traversable;
	};

	struct DefaultIlluminationEstimationLaunchParams
	{
		size_t m_size;
		IlluminationEstimationProperties m_defaultIlluminationEstimationProperties;
		LightProbe<float>* m_lightProbes;
		OptixTraversableHandle m_traversable;
	};

	struct RayMLVQRenderingLaunchParams
	{
		RayMLVQRenderingProperties m_rayMLVQRenderingProperties;
		struct {
			cudaSurfaceObject_t m_outputTexture;
			size_t m_frameId;
		} m_frame;
		struct {
			cudaTextureObject_t m_environmentalMaps[6];
			float m_lightSize = 1.0f;
			glm::vec3 m_direction = glm::vec3(0, -1, 0);
		} m_skylight;
		OptixTraversableHandle m_traversable;
	};
}
