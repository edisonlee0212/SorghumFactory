#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayTracerUtilities.cuh>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <RayDataDefinations.hpp>
namespace RayTracerFacility {
	extern "C" __constant__ DefaultRenderingLaunchParams defaultRenderingLaunchParams;
	struct DefaultRenderingRadianceRayData {
		unsigned m_hitCount;
		Random m_random;
		glm::vec3 m_energy;
		glm::vec3 m_pixelNormal;
		glm::vec3 m_pixelAlbedo;
	};
	struct DefaultRenderingSpSamplerRayData
	{
		unsigned m_instanceId;
		glm::vec3 m_p0;
		glm::vec3 m_p1;
		glm::vec3 m_n1;
	};
#pragma region Closest hit functions
	extern "C" __global__ void __closesthit__radiance()
	{
#pragma region Retrive information
		const auto& sbtData
			= *(const DefaultSbtData*)optixGetSbtDataPointer();
		const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
		const int primitiveId = optixGetPrimitiveIndex();
		const float3 rayDirectionInternal = optixGetWorldRayDirection();
		glm::vec3 rayDirection = glm::vec3(rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
		auto indices = sbtData.m_mesh.GetIndices(primitiveId);
		auto texCoord = sbtData.m_mesh.GetTexCoord(triangleBarycentricsInternal, indices);
		auto normal = sbtData.m_mesh.GetNormal(triangleBarycentricsInternal, indices);
		if (glm::dot(rayDirection, normal) > 0.0f) {
			normal = -normal;
		}
		auto tangent = sbtData.m_mesh.GetTangent(triangleBarycentricsInternal, indices);
		glm::vec3 albedoColor = sbtData.m_material.GetAlbedo(texCoord);
		sbtData.m_material.ApplyNormalTexture(normal, texCoord, triangleBarycentricsInternal, tangent);
		auto hitPoint = sbtData.m_mesh.GetPosition(triangleBarycentricsInternal, indices);
#pragma endregion
		DefaultRenderingRadianceRayData& perRayData = *GetRayDataPointer<DefaultRenderingRadianceRayData>();
		unsigned hitCount = perRayData.m_hitCount + 1;
		// start with some ambient term
		auto energy = glm::vec3(0.0f);

		uint32_t u0, u1;
		PackRayDataPointer(&perRayData, u0, u1);
		float metallic = sbtData.m_material.m_metallic;
		float roughness = sbtData.m_material.m_roughness;
		const auto scatterSamples = defaultRenderingLaunchParams.m_defaultRenderingProperties.m_samplesPerHit;
		for (int sampleID = 0; sampleID < scatterSamples; sampleID++)
		{
			perRayData.m_hitCount = hitCount;
			perRayData.m_energy = glm::vec3(0.0f);
			if (perRayData.m_hitCount <= defaultRenderingLaunchParams.m_defaultRenderingProperties.m_bounceLimit) {
				energy = glm::vec3(0.0f);
				float f = 1.0f;
				if (metallic >= 0.0f) f = (metallic + 2) / (metallic + 1);
				float3 incidentRayOrigin;
				float3 newRayDirectionInternal;
				BRDF(metallic, perRayData.m_random,
					normal, hitPoint, rayDirection,
					incidentRayOrigin, newRayDirectionInternal);
				optixTrace(defaultRenderingLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirectionInternal,
					1e-3f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					static_cast<OptixVisibilityMask>(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // SBT offset
					static_cast<int>(DefaultRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // missSBTIndex
					u0, u1);
				energy += albedoColor
					* glm::clamp(glm::abs(glm::dot(normal, glm::vec3(newRayDirectionInternal.x, newRayDirectionInternal.y, newRayDirectionInternal.z))) * roughness + (1.0f - roughness) * f, 0.0f, 1.0f)
					* perRayData.m_energy;
			}
		}
		energy /= scatterSamples;


		if (hitCount == 1) {
			perRayData.m_pixelNormal = normal;
			perRayData.m_pixelAlbedo = albedoColor;
		}
		perRayData.m_energy = energy + sbtData.m_material.m_diffuseIntensity * albedoColor;
	}
	extern "C" __global__ void __closesthit__sampleSp()
	{
		DefaultRenderingSpSamplerRayData& perRayData = *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
		assert(perRayData.m_instanceId == optixGetInstanceId());
		const auto& sbtData
			= *(const DefaultSbtData*)optixGetSbtDataPointer();
		const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
		const auto centrics = optixGetTriangleBarycentrics();
		perRayData.m_p1 = sbtData.m_mesh.GetPosition(centrics, indices);
		perRayData.m_n1 = sbtData.m_mesh.GetNormal(centrics, indices);
	}
#pragma endregion
#pragma region Any hit functions
	extern "C" __global__ void __anyhit__radiance()
	{


	}
	extern "C" __global__ void __anyhit__sampleSp()
	{
		DefaultRenderingSpSamplerRayData& perRayData = *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
		if (perRayData.m_instanceId != optixGetInstanceId()) optixIgnoreIntersection();
		const auto& sbtData
			= *(const DefaultSbtData*)optixGetSbtDataPointer();
		const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
		const auto hitPoint = sbtData.m_mesh.GetPosition(optixGetTriangleBarycentrics(), indices);
		const auto origin = optixGetWorldRayOrigin();
		const float distance = glm::distance(hitPoint, glm::vec3(origin.x, origin.y, origin.z));
		if (distance > sbtData.m_material.GetRadiusMax()) optixIgnoreIntersection();
	}
#pragma endregion
#pragma region Miss functions
	extern "C" __global__ void __miss__radiance()
	{
		DefaultRenderingRadianceRayData& prd = *GetRayDataPointer<DefaultRenderingRadianceRayData>();
		const float3 rayDir = optixGetWorldRayDirection();
		float4 environmentalLightColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		if (defaultRenderingLaunchParams.m_defaultRenderingProperties.m_useEnvironmentalMap) environmentalLightColor = SampleCubeMap<float4>(defaultRenderingLaunchParams.m_skylight.m_environmentalMaps, rayDir);
		prd.m_pixelAlbedo = prd.m_energy = glm::vec3(environmentalLightColor.x, environmentalLightColor.y, environmentalLightColor.z);
		prd.m_energy *= defaultRenderingLaunchParams.m_defaultRenderingProperties.m_skylightIntensity;
	}
	extern "C" __global__ void __miss__sampleSp()
	{

	}
#pragma endregion
#pragma region Main ray generation
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		float ix = optixGetLaunchIndex().x;
		float iy = optixGetLaunchIndex().y;
		DefaultRenderingRadianceRayData cameraRayData;
		cameraRayData.m_hitCount = 0;
		cameraRayData.m_random.Init(ix + defaultRenderingLaunchParams.m_defaultRenderingProperties.m_frameSize.x * iy,
			defaultRenderingLaunchParams.m_frame.m_frameId);
		cameraRayData.m_energy = glm::vec3(0);
		cameraRayData.m_pixelNormal = glm::vec3(0);
		cameraRayData.m_pixelAlbedo = glm::vec3(0);
		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		PackRayDataPointer(&cameraRayData, u0, u1);

		const auto numPixelSamples = defaultRenderingLaunchParams.m_defaultRenderingProperties.m_samplesPerPixel;
		auto pixelColor = glm::vec3(0.f);
		auto pixelNormal = glm::vec3(0.f);
		auto pixelAlbedo = glm::vec3(0.f);

		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
		{
			// normalized screen plane position, in [0,1]^2
			// iw: note for de-noising that's not actually correct - if we
			// assume that the camera should only(!) cover the de-noised
			// screen then the actual screen plane we should be using during
			// rendering is slightly larger than [0,1]^2
			glm::vec2 screen;
			screen = glm::vec2(ix + cameraRayData.m_random(), iy + cameraRayData.m_random()) / glm::vec2(defaultRenderingLaunchParams.m_defaultRenderingProperties.m_frameSize);
			glm::vec3 rayDir = glm::normalize(defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_direction
				+ (screen.x - 0.5f) * defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_horizontal
				+ (screen.y - 0.5f) * defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_vertical);
			float3 rayOrigin = make_float3(defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_from.x, defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_from.y, defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_from.z);
			float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

			optixTrace(defaultRenderingLaunchParams.m_traversable,
				rayOrigin,
				rayDirection,
				0.f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				static_cast<OptixVisibilityMask>(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // SBT offset
				static_cast<int>(DefaultRenderingRayType::RayTypeCount),               // SBT stride
				static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // missSBTIndex
				u0, u1);
			pixelColor += cameraRayData.m_energy;
			pixelNormal += cameraRayData.m_pixelNormal;
			pixelAlbedo += cameraRayData.m_pixelAlbedo;
			cameraRayData.m_energy = glm::vec3(0.0f);
			cameraRayData.m_pixelNormal = glm::vec3(0.0f);
			cameraRayData.m_pixelAlbedo = glm::vec3(0.0f);
			cameraRayData.m_hitCount = 0;
		}
		glm::vec3 rgb(pixelColor / static_cast<float>(numPixelSamples));

		// and write/accumulate to frame buffer ...
		if (defaultRenderingLaunchParams.m_defaultRenderingProperties.m_accumulate) {
			if (defaultRenderingLaunchParams.m_frame.m_frameId > 1) {
				float4 currentColor;
				surf2Dread(&currentColor, defaultRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
				glm::vec3 transferredCurrentColor = glm::vec4(currentColor.x, currentColor.y, currentColor.z, currentColor.w);
				rgb += static_cast<float>(defaultRenderingLaunchParams.m_frame.m_frameId) * transferredCurrentColor;
				rgb /= static_cast<float>(defaultRenderingLaunchParams.m_frame.m_frameId + 1);
			}
		}
		float4 data = make_float4(rgb.r,
			rgb.g,
			rgb.b,
			1.0f);
		// and write to frame buffer ...
		surf2Dwrite(data, defaultRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
	}
#pragma endregion
}
