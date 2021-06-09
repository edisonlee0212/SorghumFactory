#pragma once
#include <Optix7.hpp>
#include <BTFBase.cuh>
namespace RayTracerFacility
{
	struct BTFIAB : BtfBase<glm::vec3>
	{
		void Init(const std::string& parName);
	};
}