#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
#include <IndexAB.cuh>
#include <PDF1D.cuh>
namespace RayTracerFacility
{
	struct PDF2D
	{
		// the used number of 2D functions
		int m_numOfPdf2D;
		// the size of the data entry to be used here during restoration
		int m_size2D;
		// length of index slice, should be 2 here.
		int m_lengthOfSlice;
		IndexAB m_iab;
		PDF1D m_pdf1;

		
		
		// The shared coordinates to be used for interpolation
		// when retrieving the data from the database
		__device__
			virtual void GetVal(const int& pdf2DIndex, glm::vec3& out, SharedCoordinates& tc) const { out = glm::vec3(); }
	};
}