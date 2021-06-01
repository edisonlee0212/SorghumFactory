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
		DebugRendering,
		IlluminationEstimation,
		IlluminationVisualization,
		PipelineSize
	};
	
	

	template <typename LaunchParameter>
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

		LaunchParameter m_launchParams;
		CudaBuffer   m_launchParamsBuffer;
	};
	
	class OptixRayTracer
	{
	public:
		// ------------------------------------------------------------------
		// internal helper functions
		// ------------------------------------------------------------------
		[[nodiscard]] bool RenderDebugOutput(const DebugRenderingProperties& properties, std::vector<TriangleMesh>& meshes);
		void EstimateIllumination(const size_t& size, const IlluminationEstimationProperties& properties, CudaBuffer& lightProbes, std::vector<TriangleMesh>& meshes);
		OptixRayTracer();
		/*! build an acceleration structure for the given triangle mesh */
		void BuildAccelerationStructure(std::vector<TriangleMesh>& meshes);
		/*! constructs the shader binding table */
		void BuildShaderBindingTable(std::vector<TriangleMesh>& meshes, std::vector<std::pair<unsigned, cudaTextureObject_t>>& boundTextures, std::vector<cudaGraphicsResource_t>& boundResources);
		void SetAccumulate(const bool& value);
		void SetSkylightSize(const float& value);
		void SetSkylightDir(const glm::vec3& value);
		void SetStatusChanged(const bool& value = true);
	protected:
		/*! creates and configures a optix device context (in this simple
		  example, only for the primary GPU device) */
		void CreateContext();

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
		void CreatePipeline();

		
		/*! @{ CUDA device context and stream that optix pipeline will run
			on, as well as device properties for this device */
		CUcontext          m_cudaContext;
		CUstream           m_stream;
		cudaDeviceProp     m_deviceProps;
		/*! @} */

		//! the optix context that our pipeline will run in.
		OptixDeviceContext m_optixContext;

		RayTracerPipeline<DebugRenderingLaunchParams> m_debugRenderingPipeline;
		RayTracerPipeline<IlluminationEstimationLaunchParams> m_illuminationEstimationPipeline;
		
		bool m_accumulate = true;
		bool m_statusChanged = false;
		
		/*! check if we have build the acceleration structure. */
		bool m_hasAccelerationStructure = false;

		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_vertexBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_indexBuffer;
		/*! one buffer per input mesh */
		std::vector<CudaBuffer> m_vertexInfoBuffer;
		//! buffer that keeps the (final, compacted) acceleration structure
		CudaBuffer m_acceleratedStructuresBuffer;
	};

	
}
