#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayTracerUtilities.cuh>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <RayDataDefinations.hpp>

namespace RayMLVQ {
	extern "C" __constant__ RayMLVQRenderingLaunchParams rayMLVQRenderingLaunchParams;
	struct RayMLVQRenderingRayData{
		unsigned m_hitCount;
		Random m_random;
		glm::vec3 m_energy;
		glm::vec3 m_pixelNormal;
		glm::vec3 m_pixelAlbedo;
	};
#pragma region Closest hit functions
	extern "C" __global__ void __closesthit__radiance()
	{
		const auto& sbtData
			= *(const RayMLVQSbtData*)optixGetSbtDataPointer();
		const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
		const int primitiveId = optixGetPrimitiveIndex();
		const glm::uvec3 index = sbtData.m_mesh.m_triangle[primitiveId];
		const glm::vec3 pointA = sbtData.m_mesh.m_position[index.x];
		const glm::vec3 pointB = sbtData.m_mesh.m_position[index.y];
		const glm::vec3 pointC = sbtData.m_mesh.m_position[index.z];
		glm::vec3 normal;
		const float3 rayDirectionInternal = optixGetWorldRayDirection();
		glm::vec3 rayDirection = glm::vec3(rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Correct normals
		if (!rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_useGeometryNormal)
			normal = (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_mesh.m_normal[index.x]
			+ triangleBarycentricsInternal.x * sbtData.m_mesh.m_normal[index.y]
			+ triangleBarycentricsInternal.y * sbtData.m_mesh.m_normal[index.z];
#pragma endregion
		glm::vec3 albedoColor = sbtData.m_material.m_surfaceColor;
#pragma region Apply textures
		const glm::vec2 tc
			= (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_mesh.m_texCoord[index.x]
			+ triangleBarycentricsInternal.x * sbtData.m_mesh.m_texCoord[index.y]
			+ triangleBarycentricsInternal.y * sbtData.m_mesh.m_texCoord[index.z];
		if (sbtData.m_material.m_albedoTexture) {
			float4 textureAlbedo = tex2D<float4>(sbtData.m_material.m_albedoTexture, tc.x, tc.y);
			albedoColor = glm::vec3(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z);
		}
		if (sbtData.m_material.m_normalTexture)
		{
			float4 textureNormal = tex2D<float4>(sbtData.m_material.m_normalTexture, tc.x, tc.y);
			glm::vec3 tangent = (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_mesh.m_tangent[index.x]
				+ triangleBarycentricsInternal.x * sbtData.m_mesh.m_tangent[index.y]
				+ triangleBarycentricsInternal.y * sbtData.m_mesh.m_tangent[index.z];
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
		RayMLVQRenderingRayData& perRayData = *GetRayDataPointer<RayMLVQRenderingRayData>();
		unsigned hitCount = perRayData.m_hitCount + 1;
		// start with some ambient term
		auto energy = glm::vec3(0.0f);


		uint32_t u0, u1;
		PackRayDataPointer(&perRayData, u0, u1);
		float metallic = sbtData.m_material.m_metallic;
		float roughness = sbtData.m_material.m_roughness;
		const auto scatterSamples = rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_samplesPerHit;
		for (int sampleID = 0; sampleID < scatterSamples; sampleID++)
		{
			perRayData.m_hitCount = hitCount;
			perRayData.m_energy = glm::vec3(0.0f);
			if (perRayData.m_hitCount <= rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_bounceLimit) {
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
				optixTrace(rayMLVQRenderingLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirectionInternal,
					1e-3f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					static_cast<OptixVisibilityMask>(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					static_cast<int>(RayMLVQRenderingRayType::RadianceRayType),             // SBT offset
					static_cast<int>(RayMLVQRenderingRayType::RayTypeCount),               // SBT stride
					static_cast<int>(RayMLVQRenderingRayType::RadianceRayType),             // missSBTIndex
					u0, u1);
				energy += albedoColor
					* glm::clamp(glm::abs(glm::dot(normal, newRayDirection)) * roughness + (1.0f - roughness) * f, 0.0f, 1.0f)
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
#pragma endregion
#pragma region Any hit functions
	extern "C" __global__ void __anyhit__radiance()
	{
	}
	
#pragma endregion
#pragma region Miss functions
	extern "C" __global__ void __miss__radiance()
	{
		RayMLVQRenderingRayData& prd = *GetRayDataPointer<RayMLVQRenderingRayData>();
		const float3 rayDir = optixGetWorldRayDirection();
		float4 environmentalLightColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		if (rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_useEnvironmentalMap) environmentalLightColor = SampleCubeMap<float4>(rayMLVQRenderingLaunchParams.m_skylight.m_environmentalMaps, rayDir);
		prd.m_pixelAlbedo = prd.m_energy = glm::vec3(environmentalLightColor.x, environmentalLightColor.y, environmentalLightColor.z);
		prd.m_energy *= rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_skylightIntensity;
	}
	
#pragma endregion
#pragma region Main ray generation
	extern "C" __global__ void __raygen__renderFrame()
	{
		// compute a test pattern based on pixel ID
		float ix = optixGetLaunchIndex().x;
		float iy = optixGetLaunchIndex().y;
		RayMLVQRenderingRayData cameraRayData;
		cameraRayData.m_hitCount = 0;
		cameraRayData.m_random.Init(ix + rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_frameSize.x * iy,
			rayMLVQRenderingLaunchParams.m_frame.m_frameId);
		cameraRayData.m_energy = glm::vec3(0);
		cameraRayData.m_pixelNormal = glm::vec3(0);
		cameraRayData.m_pixelAlbedo = glm::vec3(0);
		// the values we store the PRD pointer in:
		uint32_t u0, u1;
		PackRayDataPointer(&cameraRayData, u0, u1);

		const auto numPixelSamples = rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_samplesPerPixel;
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
			screen = glm::vec2(ix + cameraRayData.m_random(), iy + cameraRayData.m_random()) / glm::vec2(rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_frameSize);
			glm::vec3 rayDir = glm::normalize(rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_direction
				+ (screen.x - 0.5f) * rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_horizontal
				+ (screen.y - 0.5f) * rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_vertical);
			float3 rayOrigin = make_float3(rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_from.x, rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_from.y, rayMLVQRenderingLaunchParams.m_rayMLVQRenderingProperties.m_camera.m_from.z);
			float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

			optixTrace(rayMLVQRenderingLaunchParams.m_traversable,
				rayOrigin,
				rayDirection,
				0.f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				static_cast<OptixVisibilityMask>(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				static_cast<int>(RayMLVQRenderingRayType::RadianceRayType),             // SBT offset
				static_cast<int>(RayMLVQRenderingRayType::RayTypeCount),               // SBT stride
				static_cast<int>(RayMLVQRenderingRayType::RadianceRayType),             // missSBTIndex
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
		if (rayMLVQRenderingLaunchParams.m_frame.m_frameId > 1) {
			float4 currentColor;
			surf2Dread(&currentColor, rayMLVQRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
			glm::vec3 transferredCurrentColor = glm::vec4(currentColor.x, currentColor.y, currentColor.z, currentColor.w);
			rgb += static_cast<float>(rayMLVQRenderingLaunchParams.m_frame.m_frameId) * transferredCurrentColor;
			rgb /= static_cast<float>(rayMLVQRenderingLaunchParams.m_frame.m_frameId + 1);
		}
		float4 data = make_float4(rgb.r,
			rgb.g,
			rgb.b,
			1.0f);
		// and write to frame buffer ...
		surf2Dwrite(data, rayMLVQRenderingLaunchParams.m_frame.m_outputTexture, ix * sizeof(float4), iy);
	}
#pragma endregion
}
