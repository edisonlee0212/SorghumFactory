#pragma once
#include <ray_tracer_facility_export.h>
#include <memory>
#include <RayTracer.hpp>
struct cudaGraphicsResource;
namespace RayTracerFacility {
	class RAY_TRACER_FACILITY_API CudaModule {
#pragma region Class related
		CudaModule() = default;
		CudaModule(CudaModule&&) = default;
		CudaModule(const CudaModule&) = default;
		CudaModule& operator=(CudaModule&&) = default;
		CudaModule& operator=(const CudaModule&) = default;
#pragma endregion	
		void* m_optixHandle = nullptr;
		bool m_initialized = false;
		std::unique_ptr<RayTracer> m_rayTracer;
	public:
		static std::unique_ptr<RayTracer>& GetRayTracer();
		static CudaModule& GetInstance();
		static void Init();
		static void Terminate();
		static void EstimateIlluminationRayTracing(const IlluminationEstimationProperties& properties, std::vector<LightProbe<float>>& lightProbes);
	};
}
