#include <cstdio>
#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayTracer.hpp>
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

