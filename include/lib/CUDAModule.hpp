#pragma once
#include <raymlvq_export.h>
#include <CUDABuffer.hpp>
#include <DefaultOutputRenderType.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

struct cudaGraphicsResource;
namespace RayMLVQ {
	struct RAYMLVQ_API Camera {
		bool m_modified = false;
		float m_fov = 60;
		/*! camera position - *from* where we are looking */
		glm::vec3 m_from = glm::vec3(0.0f);
		/*! which direction we are looking *at* */
		glm::vec3 m_direction = glm::vec3(0.0f);
		/*! general up-vector */
		glm::vec3 m_horizontal;
		glm::vec3 m_vertical;
		bool operator !=(const Camera& other) const {
			return
				other.m_fov != this->m_fov ||
				other.m_from != this->m_from ||
				other.m_direction != this->m_direction ||
				other.m_horizontal != this->m_horizontal ||
				other.m_vertical != this->m_vertical;
		}
		void Set(const glm::quat& rotation, const glm::vec3& position, const float& fov, const glm::ivec2& size);
	};
	
	struct RAYMLVQ_API DefaultRenderingProperties
	{
		bool m_accumulate = true;
		bool m_useEnvironmentalMap = false;
		float m_skylightIntensity = 0.8f;
		int m_bounceLimit = 4;
		int m_samplesPerPixel = 1;
		int m_samplesPerHit = 1;
		bool m_useGeometryNormal = false;
		DefaultOutputRenderType m_debugRenderingType = DefaultOutputRenderType::Brdf;
		Camera m_camera;
		unsigned m_outputTextureId;
		unsigned m_environmentalMapId;
		glm::ivec2 m_frameSize;
		[[nodiscard]] bool Changed(const DefaultRenderingProperties& properties) const
		{
			return
				properties.m_accumulate != m_accumulate ||
				properties.m_useEnvironmentalMap != m_useEnvironmentalMap ||
				properties.m_skylightIntensity != m_skylightIntensity ||
				properties.m_bounceLimit != m_bounceLimit ||
				properties.m_samplesPerPixel != m_samplesPerPixel ||
				properties.m_useGeometryNormal != m_useGeometryNormal ||
				properties.m_debugRenderingType != m_debugRenderingType ||
				properties.m_outputTextureId != m_outputTextureId ||
				properties.m_environmentalMapId != m_environmentalMapId ||
				properties.m_frameSize != m_frameSize ||
				properties.m_camera != m_camera;
		}
	};
	struct RAYMLVQ_API IlluminationEstimationProperties
	{
		unsigned m_seed = 0;
		int m_bounceLimit = 2;
		int m_numPointSamples = 100;
		int m_numScatterSamples = 10;
		float m_skylightPower = 1.0f;
		bool m_pushNormal = true;
	};

	struct RAYMLVQ_API RayMLVQRenderingProperties
	{
		bool m_accumulate = true;
		bool m_useEnvironmentalMap = false;
		float m_skylightIntensity = 0.8f;
		int m_bounceLimit = 4;
		int m_samplesPerPixel = 1;
		int m_samplesPerHit = 1;
		bool m_useGeometryNormal = false;
		Camera m_camera;
		unsigned m_outputTextureId;
		unsigned m_environmentalMapId;
		glm::ivec2 m_frameSize;
		[[nodiscard]] bool Changed(const RayMLVQRenderingProperties& properties) const
		{
			return
				properties.m_accumulate != m_accumulate ||
				properties.m_useEnvironmentalMap != m_useEnvironmentalMap ||
				properties.m_skylightIntensity != m_skylightIntensity ||
				properties.m_bounceLimit != m_bounceLimit ||
				properties.m_samplesPerPixel != m_samplesPerPixel ||
				properties.m_useGeometryNormal != m_useGeometryNormal ||
				properties.m_outputTextureId != m_outputTextureId ||
				properties.m_environmentalMapId != m_environmentalMapId ||
				properties.m_frameSize != m_frameSize ||
				properties.m_camera != m_camera;
		}
	};
	
	template<typename T>
	struct RAYMLVQ_API LightProbe
	{
		glm::vec3 m_surfaceNormal;
		/**
		 * \brief The position of the light probe.
		 */
		glm::vec3 m_position;
		/**
		 * \brief The calculated overall direction where the point received most light.
		 */
		glm::vec3 m_direction;
		/**
		 * \brief The total energy received at this point.
		 */
		T m_energy = 0;
	};
	
	struct RAYMLVQ_API TriangleMesh {
		std::vector<glm::vec3>* m_positions;
		std::vector<glm::vec3>* m_normals;
		std::vector<glm::vec3>* m_tangents;
		std::vector<glm::vec4>* m_colors;
		std::vector<glm::uvec3>* m_triangles;
		std::vector<glm::vec2>* m_texCoords;
		
		size_t m_version;
		size_t m_entityId = 0;
		size_t m_entityVersion = 0;
		glm::vec3 m_surfaceColor;
		float m_roughness;
		float m_metallic;
		bool m_removeTag = false;
		glm::mat4 m_globalTransform;

		unsigned m_albedoTexture = 0;
		unsigned m_normalTexture = 0;
		float m_diffuseIntensity = 0;

		bool m_verticesUpdateFlag = true;
		bool m_transformUpdateFlag = true;

		bool m_enableMLVQ = false;
	};
	class RayTracer;
	class RAYMLVQ_API CudaModule {
#pragma region Class related
		CudaModule() = default;
		CudaModule(CudaModule&&) = default;
		CudaModule(const CudaModule&) = default;
		CudaModule& operator=(CudaModule&&) = default;
		CudaModule& operator=(const CudaModule&) = default;
#pragma endregion	
		void* m_optixHandle = nullptr;
		bool m_initialized = false;
		CudaBuffer m_devLeaves;
		CudaBuffer m_deviceTransforms;
		CudaBuffer m_deviceDirections;
		CudaBuffer m_deviceIntensities;
		std::unique_ptr<RayTracer> m_optixRayTracer;
	public:
		static void SetStatusChanged(const bool& value = true);
		static void SetSkylightSize(const float& value);
		static void SetSkylightDir(const glm::vec3& value);
		std::vector<TriangleMesh> m_meshes;
		std::vector<LightProbe<float>> m_lightProbes;
		static CudaModule& GetInstance();
		static void Init();
		static void PrepareScene();
		static bool RenderDefault(const DefaultRenderingProperties& properties);
		static bool RenderRayMLVQ(const RayMLVQRenderingProperties& properties);
		static void Terminate();
		static void EstimateIlluminationRayTracing(const IlluminationEstimationProperties& properties, std::vector<LightProbe<float>>& lightProbes);
	};
}
