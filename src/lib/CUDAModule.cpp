#include <cstdio>
#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <OptixRayTracer.hpp>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace RayMLVQ;
void Camera::Set(const glm::quat& rotation, const glm::vec3& position, const float& fov, const glm::ivec2& size)
{
	m_from = position;
	m_direction = glm::normalize(rotation * glm::vec3(0, 0, -1));
	const float cosFovY = glm::radians(fov * 0.5f);
	const float aspect = static_cast<float>(size.x) / static_cast<float>(size.y);
	m_horizontal
		= cosFovY * aspect * glm::normalize(glm::cross(m_direction, rotation * glm::vec3(0, 1, 0)));
	m_vertical
		= cosFovY * glm::normalize(glm::cross(m_horizontal, m_direction));
}

void CudaModule::SetStatusChanged(const bool& value)
{
	GetInstance().m_optixRayTracer->SetStatusChanged(value);
}

void CudaModule::SetSkylightSize(const float& value)
{
	GetInstance().m_optixRayTracer->SetSkylightSize(value);
}

void CudaModule::SetSkylightDir(const glm::vec3& value)
{
	GetInstance().m_optixRayTracer->SetSkylightDir(value);
}

CudaModule& CudaModule::GetInstance()
{
	static CudaModule instance;
	return instance;
}

void CudaModule::Init()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	CUDA_CHECK(SetDevice(0));
	OPTIX_CHECK(optixInitWithHandle(&GetInstance().m_optixHandle));
	GetInstance().m_optixRayTracer = std::make_unique<OptixRayTracer>();
	GetInstance().m_initialized = true;
}

void CudaModule::PrepareScene()
{
	if (GetInstance().m_meshes.empty()) return;
	GetInstance().m_optixRayTracer->BuildAccelerationStructure(GetInstance().m_meshes);
}

bool CudaModule::RenderRayTracerDebugOutput(const DebugRenderingProperties& properties)
{
	auto& rayTracer = GetInstance().m_optixRayTracer;
	return rayTracer->RenderDebugOutput(properties, GetInstance().m_meshes);
}

void CudaModule::Terminate()
{
	GetInstance().m_optixRayTracer.reset();
	OPTIX_CHECK(optixUninitWithHandle(GetInstance().m_optixHandle));
	CUDA_CHECK(DeviceReset());
	GetInstance().m_initialized = false;
}


void CudaModule::EstimateIlluminationRayTracing(const IlluminationEstimationProperties& properties, std::vector<LightProbe<float>>& lightProbes)
{
	auto& cudaModule = GetInstance();
#pragma region Prepare light probes
	size_t size = lightProbes.size();
	CudaBuffer deviceLightProbes;
	deviceLightProbes.Upload(lightProbes);
#pragma endregion
	cudaModule.m_optixRayTracer->EstimateIllumination(size, properties, deviceLightProbes, cudaModule.m_meshes);
	deviceLightProbes.Download(lightProbes.data(), size);
	deviceLightProbes.Free();
}