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

__global__ void IlluminationResetKernel(int size, glm::vec3* directions, float* illuminations)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		directions[idx] = glm::vec3(0, 1, 0);
		illuminations[idx] = 1.0f;
	}
}

template<typename T>
__global__ void LightProbeResetKernel(unsigned size, LightProbe<T>* lightProbes)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		lightProbes[idx].m_direction = glm::vec3(0);
		lightProbes[idx].m_energy = T();
	}
}

__global__ void IlluminationEstimationKernel(float angleFactor, float factor, int i, int size, glm::mat4* transforms, glm::vec3* directions, float* illuminations)
{

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		const glm::vec3 receiverPosition = transforms[idx][3];
		const glm::vec3 blockerPosition = transforms[i][3];
		if (receiverPosition.y < blockerPosition.y) {
			const glm::vec3 vector = receiverPosition - blockerPosition;
			const float distance2 = vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
			const float intensity = glm::pow(glm::dot(vector, glm::vec3(0, 1, 0)), angleFactor);
			if (illuminations[idx] > 0) {
				illuminations[idx] -= factor * intensity / distance2;
				if (illuminations[idx] < 0) illuminations[idx] = 0;
			}
		}
	}
}

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

void CudaModule::EstimateIlluminationShadowCone(const float& angleFactor, const float& factor, const int& size, glm::mat4* transforms, std::vector<glm::vec3>& directions, std::vector<float>& illuminations)
{
	auto& cudaModule = GetInstance();
#pragma region Memory
	cudaModule.m_deviceDirections.Resize(size * sizeof(glm::vec3));
	cudaModule.m_deviceIntensities.Resize(size * sizeof(float));
	cudaModule.m_deviceTransforms.Upload(transforms, size);
#pragma endregion


	int blockSize = 0;      // The launch configurator returned block size 
	int minGridSize = 0;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize = 0;       // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, IlluminationEstimationKernel, 0, size);
	gridSize = (size + blockSize - 1) / blockSize;
	IlluminationResetKernel << <gridSize, blockSize >> > (size, static_cast<glm::vec3*>(cudaModule.m_deviceDirections.m_dPtr), static_cast<float*>(cudaModule.m_deviceIntensities.m_dPtr));
	CUDA_SYNC_CHECK();
	for (int i = 0; i < size; i++) {
		IlluminationEstimationKernel << <gridSize, blockSize >> > (angleFactor, factor, i, size, static_cast<glm::mat4*>(cudaModule.m_deviceTransforms.m_dPtr), static_cast<glm::vec3*>(cudaModule.m_deviceDirections.m_dPtr), static_cast<float*>(cudaModule.m_deviceIntensities.m_dPtr));
	}
#pragma region Error check & sync
	// Check for any errors launching the kernel
	CUDA_ERROR_CHECK();
	CUDA_SYNC_CHECK();
#pragma endregion
	directions.resize(size);
	illuminations.resize(size);
	cudaModule.m_deviceDirections.Download(directions.data(), size);
	cudaModule.m_deviceIntensities.Download(illuminations.data(), size);
	cudaModule.m_deviceTransforms.Free();
	cudaModule.m_deviceDirections.Free();
	cudaModule.m_deviceIntensities.Free();
}

