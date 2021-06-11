#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <IndexAB.cuh>
#include <PDF1D.cuh>
namespace RayTracerFacility
{
	struct PDF2D
	{
		
		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		__device__
			virtual void GetVal(const int& pdf2DIndex, glm::vec3& out, SharedCoordinates& tc) const { out = glm::vec3(); }
	};
}