#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
namespace RayTracerFacility
{
	template<typename T>
	struct PDF2D
	{
		// the number of allocated 2D functions to be stored
		int m_maxPdf2D;
		// the used number of 2D functions
		int m_numOfPdf2D;
		// the number of indices in parameter alpha
		int m_numOfSlicesPerHemisphere;
		// the size of the data entry to be used here during restoration
		int m_size2D;
		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		__device__
			virtual void GetVal(const int& pdf2DIndex, T& out, SharedCoordinates& tc) const { out = T(); }
	};
}