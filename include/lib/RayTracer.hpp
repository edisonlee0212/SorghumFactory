#pragma once
#include <Optix7.hpp>
#include <CUDABuffer.hpp>
#include <LaunchParams.hpp>
#include <CUDAModule.hpp>
#include <cuda.h>
#include <vector>
#include <glm/glm.hpp>
namespace RayMLVQ {
	enum PipelineType
	{
		DefaultRendering,
		IlluminationEstimation,
		RayMLVQRendering,
		
		PipelineSize
	};
	
	struct RayTracerPipeline
	{
		OptixModule                 m_module;
		OptixModuleCompileOptions   m_moduleCompileOptions = {};

		OptixPipeline               m_pipeline;
		OptixPipelineCompileOptions m_pipelineCompileOptions = {};
		OptixPipelineLinkOptions    m_pipelineLinkOptions = {};

		std::vector<OptixProgramGroup> m_rayGenProgramGroups;
		CudaBuffer m_rayGenRecordsBuffer;
		std::vector<OptixProgramGroup> m_missProgramGroups;
		CudaBuffer m_missRecordsBuffer;
		std::vector<OptixProgramGroup> m_hitGroupProgramGroups;
		CudaBuffer m_hitGroupRecordsBuffer;
		OptixShaderBindingTable m_sbt = {};
	};

	class RayTracer
	{
	public:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------
		[[nodiscard]] bool RenderDebugOutput(const DefaultRenderingProperties& properties, std::vector<TriangleMesh>& meshes);
		void EstimateIllumination(const size_t& size, const IlluminationEstimationProperties& properties, CudaBuffer& lightProbes, std::vector<TriangleMesh>& meshes);
		RayTracer();
		/*! build an acceleration structure for the given triangle mesh */
		void BuildAccelerationStructure(std::vector<TriangleMesh>& meshes);
		/*! constructs the shader binding table */
		void BuildShaderBindingTable(std::vector<TriangleMesh>& meshes, std::vector<std::pair<unsigned, cudaTextureObject_t>>& boundTextures, std::vector<cudaGraphicsResource_t>& boundResources);
		void SetAccumulate(const bool& value);
		void SetSkylightSize(const float& value);
		void SetSkylightDir(const glm::vec3& value);
		void SetStatusChanged(const bool& value = true);
	protected:
#pragma region Device and context
		/*! @{ CUDA device context and stream that optix pipeline will run
			on, as well as device properties for this device */
		CUcontext          m_cudaContext;
		CUstream           m_stream;
		cudaDeviceProp     m_deviceProps;
		/*! @} */
		//! the optix context that our pipeline will run in.
		OptixDeviceContext m_optixContext;

		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void CreateContext();
#pragma endregion
#pragma region Pipeline setup

		DefaultRenderingLaunchParams m_defaultRenderingLaunchParams;
		DefaultIlluminationEstimationLaunchParams m_defaultIlluminationEstimationLaunchParams;
		RayMLVQRenderingLaunchParams m_rayMLVQRenderingLaunchParams;

		CudaBuffer m_defaultRenderingLaunchParamsBuffer;
		CudaBuffer m_defaultIlluminationEstimationLaunchParamsBuffer;
		CudaBuffer m_rayMLVQRenderingLaunchParamsBuffer;
		
		RayTracerPipeline m_defaultRenderingPipeline;
		RayTracerPipeline m_defaultIlluminationEstimationPipeline;
		RayTracerPipeline m_rayMLVQRenderingPipeline;
		
		/*! creates the module that contains all the programs we are going
		  to use. in this simple example, we use a single module from a
		  single .cu file, using a single embedded ptx string */
		void CreateModule();

		/*! does all setup for the rayGen program(s) we are going to use */
		void CreateRayGenPrograms();

		/*! does all setup for the miss program(s) we are going to use */
		void CreateMissPrograms();

		/*! does all setup for the hitGroup program(s) we are going to use */
		void CreateHitGroupPrograms();

		/*! assembles the full pipeline of all programs */
		void AssemblePipelines();

		void AssemblePipeline(RayTracerPipeline& targetPipeline);
		
#pragma endregion

		bool m_accumulate = true;
		bool m_statusChanged = false;
#pragma region Accleration structure
		/*! check if we have build the acceleration structure. */
		bool m_hasAccelerationStructure = false;

		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_positionsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_normalsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_tangentsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_transformedPositionsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_transformedNormalsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_transformedTangentsBuffer;

		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_trianglesBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_texCoordsBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_colorsBuffer;
		//! buffer that keeps the (final, compacted) acceleration structure
		CudaBuffer m_acceleratedStructuresBuffer;
#pragma endregion
	};


	
}
