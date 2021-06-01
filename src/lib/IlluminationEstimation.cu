#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <LaunchParams.hpp>
#include <RayTracerUtilities.cuh>
namespace Cuda {
	extern "C" __constant__ IlluminationEstimationLaunchParams illuminationEstimationLaunchParams;
	struct IlluminationEstimationRayData {
		unsigned m_hitCount = 0;
		Random m_random;
		float m_energy = 0;
	};
#pragma region Closest hit functions
	extern "C" __global__ void __closesthit__illuminationEstimation()
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
#pragma region Normals
		normal = (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_vertexInfo[index.x].m_normal
			+ triangleBarycentricsInternal.x * sbtData.m_vertexInfo[index.y].m_normal
			+ triangleBarycentricsInternal.y * sbtData.m_vertexInfo[index.z].m_normal;
#pragma endregion
		//glm::vec3 albedoColor = sbtData.m_color;
#pragma region Apply textures
		const glm::vec2 tc
			= (1.f - triangleBarycentricsInternal.x - triangleBarycentricsInternal.y) * sbtData.m_vertexInfo[index.x].m_texCoords
			+ triangleBarycentricsInternal.x * sbtData.m_vertexInfo[index.y].m_texCoords
			+ triangleBarycentricsInternal.y * sbtData.m_vertexInfo[index.z].m_texCoords;
		/*
		if (sbtData.m_albedoTexture) {
			float4 textureAlbedo = tex2D<float4>(sbtData.m_albedoTexture, tc.x, tc.y);
			albedoColor = glm::vec3(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z);
		}
		*/
		if (sbtData.m_normalTexture)
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
		
		IlluminationEstimationRayData& perRayData = *GetRayDataPointer<IlluminationEstimationRayData>();
		auto energy = 0.0f;
		const auto scatterSamples = illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_numScatterSamples;
		uint32_t u0, u1;
		PackRayDataPointer(&perRayData, u0, u1);
		float metallic = sbtData.m_metallic;
		float roughness = sbtData.m_roughness;
		unsigned hitCount = perRayData.m_hitCount + 1;
		for (int sampleID = 0; sampleID < scatterSamples; sampleID++)
		{
			perRayData.m_hitCount = hitCount;
			perRayData.m_energy = 0;
			if (perRayData.m_hitCount <= illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_bounceLimit) {
				float f = 1.0f;
				if (metallic >= 0.0f) f = (metallic + 2) / (metallic + 1);
				glm::vec3 reflected = Reflect(rayDirection, normal);
				glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random, reflected, metallic);
				auto origin = hitPoint;
				if(glm::dot(newRayDirection, normal) > 0.0f)
				{
					origin += normal * 1e-3f;
				}else
				{
					origin -= normal * 1e-3f;
				}
				float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
				float3 newRayDirectionInternal = make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
				optixTrace(illuminationEstimationLaunchParams.m_traversable,
					incidentRayOrigin,
					newRayDirectionInternal,
					0.0f,    // tmin
					1e20f,  // tmax
					0.0f,   // rayTime
					static_cast<OptixVisibilityMask>(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
					static_cast<int>(IlluminationEstimationRayType::RadianceRayType),             // SBT offset
					static_cast<int>(IlluminationEstimationRayType::RayTypeCount),               // SBT stride
					static_cast<int>(IlluminationEstimationRayType::RadianceRayType),             // missSBTIndex
					u0, u1);
				energy += glm::clamp(glm::abs(glm::dot(normal, newRayDirection)) * roughness + (1.0f - roughness) * f, 0.0f, 1.0f)
					* perRayData.m_energy;
			}
		}
		perRayData.m_energy = energy / scatterSamples;
	}
#pragma endregion
#pragma region Any hit functions
	extern "C" __global__ void __anyhit__illuminationEstimation()
	{

	}
#pragma endregion
#pragma region Miss functions
	extern "C" __global__ void __miss__illuminationEstimation()
	{
		IlluminationEstimationRayData& prd = *GetRayDataPointer<IlluminationEstimationRayData>();
		const float3 rayDirectionInternal = optixGetWorldRayDirection();
		const glm::vec3 rayDirection = glm::vec3(rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
		prd.m_energy = illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_skylightPower;
	}
#pragma endregion
#pragma region Main ray generation
	extern "C" __global__ void __raygen__illuminationEstimation()
	{
		unsigned ix = optixGetLaunchIndex().x;
		const auto numPointSamples = illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_numPointSamples;
		const auto position = illuminationEstimationLaunchParams.m_lightProbes[ix].m_position;
		const auto surfaceNormal = illuminationEstimationLaunchParams.m_lightProbes[ix].m_surfaceNormal;
		const bool pushNormal = illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_pushNormal;
		float pointEnergy = 0.0f;
		auto pointDirection = glm::vec3(0.0f);

		IlluminationEstimationRayData perRayData;
		perRayData.m_random.Init(ix, illuminationEstimationLaunchParams.m_illuminationEstimationProperties.m_seed);
		uint32_t u0, u1;
		PackRayDataPointer(&perRayData, u0, u1);
		for (int sampleID = 0; sampleID < numPointSamples; sampleID++)
		{
			perRayData.m_energy = 0.0f;
			perRayData.m_hitCount = 0;
			glm::vec3 rayDir = RandomSampleHemisphere(perRayData.m_random, surfaceNormal, 0.0f);
			glm::vec3 rayOrigin = position;
			if (pushNormal) {
				if (glm::dot(rayDir, surfaceNormal) > 0)
				{
					rayOrigin += surfaceNormal * 1e-3f;
				}
				else
				{
					rayOrigin -= surfaceNormal * 1e-3f;
				}
			}
			float3 rayOriginInternal = make_float3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
			float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);
			optixTrace(illuminationEstimationLaunchParams.m_traversable,
				rayOriginInternal,
				rayDirection,
				1e-2f,    // tmin
				1e20f,  // tmax
				0.0f,   // rayTime
				static_cast<OptixVisibilityMask>(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
				static_cast<int>(IlluminationEstimationRayType::RadianceRayType),             // SBT offset
				static_cast<int>(IlluminationEstimationRayType::RayTypeCount),               // SBT stride
				static_cast<int>(IlluminationEstimationRayType::RadianceRayType),             // missSBTIndex
				u0, u1);
			pointEnergy += perRayData.m_energy;
			pointDirection += rayDir * perRayData.m_energy;
		}
		if (pointEnergy != 0) {
			illuminationEstimationLaunchParams.m_lightProbes[ix].m_energy = pointEnergy / numPointSamples;
			illuminationEstimationLaunchParams.m_lightProbes[ix].m_direction = pointDirection / static_cast<float>(numPointSamples);
		}
	}
#pragma endregion
}
