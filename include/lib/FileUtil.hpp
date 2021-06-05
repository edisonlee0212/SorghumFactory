#pragma once
#include <ray_tracer_facility_export.h>
#include <sstream>
namespace RayTracerFacility {
	class RAY_TRACER_FACILITY_API FileIO
	{
	public:
		static std::string LoadFileAsString(const std::string& path = "");
	};
}
