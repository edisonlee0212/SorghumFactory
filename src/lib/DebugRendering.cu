#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <LaunchParams.hpp>
#include <RayTracerUtilities.cuh>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
namespace RayMLVQ {
	extern "C" __constant__ DebugRenderingLaunchParams debugRenderingLaunchParams;
	struct DebugRenderingRayData {
		unsigned m_hitCount;
		Random m_random;
		glm::vec3 m_energy;
		glm::vec3 m_pixelNormal;
		glm::vec3 m_pixelAlbedo;
	};
#pragma region Closest hit functions
	extern "C" __global__ void __closesthit__shadow()
	{

	}
	extern "C" __global__ void __closesthit__radiance()
	{
		const auto& sbtData
			= *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
		const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
		const int primitiveId = optixGetPrimitiveIndex();
		const glm::ivec3 index = sbtData.m_index[primitiveId];
		const glm::vec3 pointA = sbtData.m_vertex[index.x];
		const glm::vec3 pointB = sbtData.m_vertex[index.y];
		const glm::vec3 pointC = sbtData.m_vertex[index.z];
		glm::vec3 normal;
		const float3 rayDirectionInternal = optixGetWorldRayDirection();
		glm::vec3 rayDirection = glm::vec3(rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Correct normals
		if (!debugRenderingLaunchParams.m_debugRenderingProperties.m_useGeometryNormal)
			normal = (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_vertexInfo[index.x].m_normal
			+ triangleBarycentricsInternal.x * sbtData.m_vertexInfo[index.y].m_normal
			+ triangleBarycentricsInternal.y * sbtData.m_vertexInfo[index.z].m_normal;
		/*
		if (glm::dot(rayDirection, normal) > 0.f) {
			normal = -normal;
		}
		normal = glm::normalize(normal);
		*/
#pragma endregion
		glm::vec3 albedoColor = sbtData.m_color;
#pragma region Apply textures
		const glm::vec2 tc
			= (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_vertexInfo[index.x].m_texCoords
			+ triangleBarycentricsInternal.x * sbtData.m_vertexInfo[index.y].m_texCoords
			+ triangleBarycentricsInternal.y * sbtData.m_vertexInfo[index.z].m_texCoords;
		if (sbtData.m_albedoTexture) {
			float4 textureAlbedo = tex2D<float4>(sbtData.m_albedoTexture, tc.x, tc.y);
			albedoColor = glm::vec3(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z);
		}
		if(sbtData.m_normalTexture)
		{
			float4 textureNormal = tex2D<float4>(sbtData.m_normalTexture, tc.x, tc.y);
			glm::vec3 tangent = (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_vertexInfo[index.x].m_tangent
				+ triangleBarycentricsInternal.x * sbtData.m_vertexInfo[index.y].m_tangent
				+ triangleBarycentricsInternal.y * sbtData.m_vertexInfo[index.z].m_tangent;
			glm::vec3 B = glm::cross(normal, tangent);
			glm::mat3 TBN = glm::mat3(tangent, B, normal);
			normal = glm::vec3(textureNormal.x, textureNormal.y, textureNormal.z) * 2.0f - glm::vec3(1.0f);
			normal = glm::normalize(TBN * normal);
		}
#pragma endregion
		const glm::vec3 hitPoint
			= (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * pointA
			+ triangleBarycentricsInternal.x * pointB
			+ triangleBarycentricsInternal.y * pointC;
		DebugRenderingRayData& perRayData = *GetRayDataPointer<DebugRenderingRayData>();
		unsigned hitCount = perRayData.m_hitCount + 1;
		// start with some ambient term
		auto energy = glm::vec3(0.0f);
		switch (debugRenderingLaunchParams.m_debugRenderingProperties.m_debugRenderingType)
		{
		case DebugOutputRenderType::SoftShadow:
		{
			energy += glm::vec3(0.1f) + 0.2f * fabsf(glm::dot(normal, rayDirection)) * albedoColor;
			// produce random light sample
			const float lightSize = debugRenderingLaunchParams.m_skylight.m_lightSize;
			glm::vec3 lightDir = -debugRenderingLaunchParams.m_skylight.m_direction + glm::vec3(perRayData.m_random() - 0.5f, perRayData.m_random() - 0.5f, perRayData.m_random() - 0.5f) * lightSize;
			auto origin = hitPoint + 1e-3f * normal;
			float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
			// trace shadow ray:
			const float NdotL = dot(lightDir, normal);
			if (NdotL >= 0.f) {
				auto shadowRayData = glm::vec3(0);
				// the values we store the PRD pointer in:
				uint32_t u0, u1;
				PackRayDataPointer(&shadowRayData, u0, u1);
				float3 newRayDirection = make_float3(lightDir.x, lightDir.y, lightDir.z);
				optixTrace(debugRenderingLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirection,
					1e-3f,      // tmin
					1e20f,  // tmax
					0.0f,       // rayTime
					static_cast<OptixVisibilityMask>(255),
					// For shadow rays: skip any/closest hit shaders and terminate on first
					// intersection with anything. The miss shader is used to mark if the
					// light was visible.
					OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
					| OPTIX_RAY_FLAG_DISABLE_ANYHIT
					| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
					| OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					static_cast<int>(DebugRenderingRayType::ShadowRayType),            // SBT offset
					static_cast<int>(DebugRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(DebugRenderingRayType::ShadowRayType),            // missSBTIndex 
					u0, u1);
				energy
					+= shadowRayData
					* debugRenderingLaunchParams.m_debugRenderingProperties.m_skylightIntensity
					* albedoColor
					* NdotL;

			}
		}
		break;
		case DebugOutputRenderType::Glass:
		{
			perRayData.m_hitCount = hitCount;
			if (perRayData.m_hitCount <= debugRenderingLaunchParams.m_debugRenderingProperties.m_bounceLimit) {
				uint32_t u0, u1;
				PackRayDataPointer(&perRayData, u0, u1);
				glm::vec3 newRayDirection = Reflect(rayDirection, normal);
				auto origin = hitPoint;
				if (glm::dot(newRayDirection, normal) > 0.0f)
				{
					origin += normal * 1e-3f;
				}
				else
				{
					origin -= normal * 1e-3f;
				}
				float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
				float3 newRayDirectionInternal = make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
				optixTrace(debugRenderingLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirectionInternal,
					1e-3f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					static_cast<OptixVisibilityMask>(255),
					OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES | OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					static_cast<int>(DebugRenderingRayType::RadianceRayType),             // SBT offset
					static_cast<int>(DebugRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(DebugRenderingRayType::RadianceRayType),             // missSBTIndex
					u0, u1);
				const auto cos = glm::clamp(glm::abs(glm::dot(normal, newRayDirection)) * sbtData.m_roughness + (1.0f - sbtData.m_roughness), 0.0f, 1.0f);
				energy += cos * perRayData.m_energy;
			}
		}
		break;
		case DebugOutputRenderType::Brdf:
		{
			uint32_t u0, u1;
			PackRayDataPointer(&perRayData, u0, u1);
			float metallic = sbtData.m_metallic;
			float roughness = sbtData.m_roughness;
			const auto scatterSamples = debugRenderingLaunchParams.m_debugRenderingProperties.m_samplesPerHit;
			for (int sampleID = 0; sampleID < scatterSamples; sampleID++)
			{
				perRayData.m_hitCount = hitCount;
				perRayData.m_energy = glm::vec3(0.0f);
				if (perRayData.m_hitCount <= debugRenderingLaunchParams.m_debugRenderingProperties.m_bounceLimit) {
					energy = glm::vec3(0.0f);
					float f = 1.0f;
					if (metallic >= 0.0f) f = (metallic + 2) / (metallic + 1);
					glm::vec3 reflected = Reflect(rayDirection, normal);
					glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random, reflected, metallic);
					auto origin = hitPoint;
					if (glm::dot(newRayDirection, normal) > 0.0f)
					{
						origin += normal * 1e-3f;
					}
					else
					{
						origin -= normal * 1e-3f;
					}
					float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
					float3 newRayDirectionInternal = make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
					optixTrace(debugRenderingLaunchParams.m_traversable,
						incidentRayOrigin,
						newRayDirectionInternal,
						1e-3f,    // tmin
						1e20f,  // tmax
						0.0f,   // rayTime
						static_cast<OptixVisibilityMask>(255),
						OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
						static_cast<int>(DebugRenderingRayType::RadianceRayType),             // SBT offset
						static_cast<int>(DebugRenderingRayType::RayTypeCount),               // SBT stride
						static_cast<int>(DebugRenderingRayType::RadianceRayType),             // missSBTIndex
						u0, u1);
					energy += albedoColor
						* glm::clamp(glm::abs(glm::dot(normal, newRayDirection)) * roughness + (1.0f - roughness) * f, 0.0f, 1.0f)
						* perRayData.m_energy;
				}
			}
			energy /= scatterSamples;
		}
		break;
		}
		if (hitCount == 1) {
			perRayData.m_pixelNormal = normal;
			perRayData.m_pixelAlbedo = albedoColor;
		}
		perRayData.m_energy = energy + sbtData.m_diffuseIntensity * albedoColor;
	}
#pragma endregion
#pragma region Any hit functions
	extern "C" __global__ void __anyhit__radiance()
	{
	}

	extern "C" __global__ void __anyhit__shadow()
	{
	}
#pragma endregion
#pragma region Miss functions
	extern "C" __global__ void __miss__radiance()
	{
		DebugRenderingRayData& prd = *GetRayDataPointer<DebugRenderingRayData>();
		const float3 rayDir = optixGetWorldRayDirection();
		float4 environmentalLightColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		if (debugRenderingLaunchParams.m_debugRenderingProperties.m_useEnvironmentalMap) environmentalLightColor = SampleCubeMap<float4>(debugRenderingLaunchParams.m_skylight.m_environmentalMaps, rayDir);
		prd.m_pixelAlbedo = prd.m_energy = glm::vec3(environmentalLightColor.x, environmentalLightColor.y, environmentalLightColor.z);
		prd.m_energy *= debugRenderingLaunchParams.m_debugRenderingProperties.m_skylightIntensity;
	}
	extern "C" __global__ void __miss__shadow()
	{
		// we didn't hit anything, so the light is visible
		glm::vec3& prd = *(glm::vec3*)GetRayDataPointer<glm::vec3>();
		prd = glm::vec3(1.0f);
	}

#pragma endregion
#pragma region Main ray generation
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		float ix = optixGetLaunchIndex().x;
		float iy = optixGetLaunchIndex().y;
		DebugRenderingRayData cameraRayData;
		cameraRayData.m_hitCount = 0;
		cameraRayData.m_random.Init(ix + debugRenderingLaunchParams.m_debugRenderingProperties.m_frameSize.x * iy,
			debugRenderingLaunchParams.m_frame.m_frameId);
		cameraRayData.m_energy = glm::vec3(0);
		cameraRayData.m_pixelNormal = glm::vec3(0);
		cameraRayData.m_pixelAlbedo = glm::vec3(0);
		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		PackRayDataPointer(&cameraRayData, u0, u1);

		const auto numPixelSamples = debugRenderingLaunchParams.m_debugRenderingProperties.m_samplesPerPixel;
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
			screen = glm::vec2(ix + cameraRayData.m_random(), iy + cameraRayData.m_random()) / glm::vec2(debugRenderingLaunchParams.m_debugRenderingProperties.m_frameSize);
			glm::vec3 rayDir = glm::normalize(debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_direction
				+ (screen.x - 0.5f) * debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_horizontal
				+ (screen.y - 0.5f) * debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_vertical);
			float3 rayOrigin = make_float3(debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_from.x, debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_from.y, debugRenderingLaunchParams.m_debugRenderingProperties.m_camera.m_from.z);
			float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

			optixTrace(debugRenderingLaunchParams.m_traversable,
				rayOrigin,
				rayDirection,
				0.f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				static_cast<OptixVisibilityMask>(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				static_cast<int>(DebugRenderingRayType::RadianceRayType),             // SBT offset
				static_cast<int>(DebugRenderingRayType::RayTypeCount),               // SBT stride
				static_cast<int>(DebugRenderingRayType::RadianceRayType),             // missSBTIndex
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
		if (debugRenderingLaunchParams.m_frame.m_frameId > 1) {
			float4 currentColor;
			surf2Dread(&currentColor, debugRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
			glm::vec3 transferredCurrentColor = glm::vec4(currentColor.x, currentColor.y, currentColor.z, currentColor.w);
			rgb += static_cast<float>(debugRenderingLaunchParams.m_frame.m_frameId) * transferredCurrentColor;
			rgb /= static_cast<float>(debugRenderingLaunchParams.m_frame.m_frameId + 1);
		}
		float4 data = make_float4(rgb.r,
			rgb.g,
			rgb.b,
			1.0f);
		// and write to frame buffer ...
		surf2Dwrite(data, debugRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
	}
#pragma endregion
}
