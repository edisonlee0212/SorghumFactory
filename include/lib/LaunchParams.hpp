#pragma once
#include <CUDAModule.hpp>
#include <glm/glm.hpp>

namespace RayMLVQ {
	enum class DebugRenderingRayType
	{
		RadianceRayType,
		ShadowRayType,
		RayTypeCount
	};
	enum class IlluminationEstimationRayType
	{
		RadianceRayType,
		RayTypeCount
	};
	struct VertexInfo;
	struct TriangleMeshSBTData {
		glm::vec3  m_color;
		glm::vec3* m_vertex;
		glm::ivec3* m_index;
		float m_roughness = 15;
		float m_metallic = 0.5;
		VertexInfo* m_vertexInfo;
		cudaTextureObject_t m_albedoTexture;
		cudaTextureObject_t m_normalTexture;
		float m_diffuseIntensity;
	};

	struct DebugRenderingLaunchParams
	{
		DebugRenderingProperties m_debugRenderingProperties;
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

	struct IlluminationEstimationLaunchParams
	{
		size_t m_size;
		IlluminationEstimationProperties m_illuminationEstimationProperties;
		LightProbe<float>* m_lightProbes;
		OptixTraversableHandle m_traversable;
	};

	struct IlluminationVisualizationLaunchParams
	{
		int m_bounceLimit = 4;
	};
}
