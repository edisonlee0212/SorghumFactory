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
	struct DefaultRenderingRayData {
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
			= *(const DefaultSbtData*)optixGetSbtDataPointer();
		const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
		const int primitiveId = optixGetPrimitiveIndex();
		const float3 rayDirectionInternal = optixGetWorldRayDirection();
		glm::vec3 rayDirection = glm::vec3(rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
		auto indices = sbtData.m_mesh.GetIndices(primitiveId);
		auto texCoord = sbtData.m_mesh.GetTexCoord(triangleBarycentricsInternal, indices);
		auto normal = sbtData.m_mesh.GetNormal(triangleBarycentricsInternal, indices);
		/*
		if (glm::dot(rayDirection, normal) > 0.f) {
			normal = -normal;
		}
		normal = glm::normalize(normal);
		*/
		auto tangent = sbtData.m_mesh.GetTangent(triangleBarycentricsInternal, indices);
		glm::vec3 albedoColor = sbtData.m_material.GetAlbedo(texCoord);
		sbtData.m_material.ApplyNormalTexture(normal, texCoord, triangleBarycentricsInternal, tangent);

		auto hitPoint = sbtData.m_mesh.GetPosition(triangleBarycentricsInternal, indices);
		
		DefaultRenderingRayData& perRayData = *GetRayDataPointer<DefaultRenderingRayData>();
		unsigned hitCount = perRayData.m_hitCount + 1;
		// start with some ambient term
		auto energy = glm::vec3(0.0f);
		switch (defaultRenderingLaunchParams.m_defaultRenderingProperties.m_debugRenderingType)
		{
		case DefaultOutputRenderType::SoftShadow:
		{
			energy += glm::vec3(0.1f) + 0.2f * fabsf(glm::dot(normal, rayDirection)) * albedoColor;
			// produce random light sample
			const float lightSize = defaultRenderingLaunchParams.m_skylight.m_lightSize;
			glm::vec3 lightDir = -defaultRenderingLaunchParams.m_skylight.m_direction + glm::vec3(perRayData.m_random() - 0.5f, perRayData.m_random() - 0.5f, perRayData.m_random() - 0.5f) * lightSize;
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
				optixTrace(defaultRenderingLaunchParams.m_traversable,
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
					static_cast<int>(DefaultRenderingRayType::ShadowRayType),            // SBT offset
					static_cast<int>(DefaultRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(DefaultRenderingRayType::ShadowRayType),            // missSBTIndex 
					u0, u1);
				energy
					+= shadowRayData
					* defaultRenderingLaunchParams.m_defaultRenderingProperties.m_skylightIntensity
					* albedoColor
					* NdotL;

			}
		}
		break;
		case DefaultOutputRenderType::Glass:
		{
			perRayData.m_hitCount = hitCount;
			if (perRayData.m_hitCount <= defaultRenderingLaunchParams.m_defaultRenderingProperties.m_bounceLimit) {
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
				optixTrace(defaultRenderingLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirectionInternal,
					1e-3f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					static_cast<OptixVisibilityMask>(255),
					OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES | OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // SBT offset
					static_cast<int>(DefaultRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(DefaultRenderingRayType::RadianceRayType),             // missSBTIndex
					u0, u1);
				const auto cos = glm::clamp(glm::abs(glm::dot(normal, newRayDirection)) * sbtData.m_material.m_roughness + (1.0f - sbtData.m_material.m_roughness), 0.0f, 1.0f);
				energy += cos * perRayData.m_energy;
			}
		}
		break;
		case DefaultOutputRenderType::Brdf:
		{
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
		perRayData.m_energy = energy + sbtData.m_material.m_diffuseIntensity * albedoColor;
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
		DefaultRenderingRayData& prd = *GetRayDataPointer<DefaultRenderingRayData>();
		const float3 rayDir = optixGetWorldRayDirection();
		float4 environmentalLightColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		if (defaultRenderingLaunchParams.m_defaultRenderingProperties.m_useEnvironmentalMap) environmentalLightColor = SampleCubeMap<float4>(defaultRenderingLaunchParams.m_skylight.m_environmentalMaps, rayDir);
		prd.m_pixelAlbedo = prd.m_energy = glm::vec3(environmentalLightColor.x, environmentalLightColor.y, environmentalLightColor.z);
		prd.m_energy *= defaultRenderingLaunchParams.m_defaultRenderingProperties.m_skylightIntensity;
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
		DefaultRenderingRayData cameraRayData;
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
		if (defaultRenderingLaunchParams.m_frame.m_frameId > 1) {
			float4 currentColor;
			surf2Dread(&currentColor, defaultRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
			glm::vec3 transferredCurrentColor = glm::vec4(currentColor.x, currentColor.y, currentColor.z, currentColor.w);
			rgb += static_cast<float>(defaultRenderingLaunchParams.m_frame.m_frameId) * transferredCurrentColor;
			rgb /= static_cast<float>(defaultRenderingLaunchParams.m_frame.m_frameId + 1);
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
