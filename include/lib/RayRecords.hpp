#pragma once
#include <LaunchParams.hpp>
#include <Optix7.hpp>
namespace RayMLVQ
{
	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DebugRenderingRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DebugRenderingRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DebugRenderingRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		TriangleMeshSBTData m_data;
	};

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationEstimationRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationEstimationRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationEstimationRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		TriangleMeshSBTData m_data;
	};

	/*! SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationVisualizationRayGenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationVisualizationRayMissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		void* m_data;
	};

	/*! SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IlluminationVisualizationRayHitRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		// just a dummy value - later examples will use more interesting
		// data here
		TriangleMeshSBTData m_data;
	};
}
