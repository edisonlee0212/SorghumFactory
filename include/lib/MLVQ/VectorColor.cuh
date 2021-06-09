#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
namespace RayTracerFacility
{
	struct VectorColor
	{
		// the index from which we start the search 
		int m_startIndex;
		// no. of channels describing one color (in our case usually 2 (CIE a-b))
		int m_numOfChannels;
		// the number of allocated a-b colors
		int m_maxVectorColor;
		int m_initialMaxVectorColor;
		// the data array of a-b colors
		float* m_vectorColorBasis;
		// current number of stored a-b colors
		int m_noOfColors;

		
		__device__
		float Get(const int& colorIndex, const int& posAB, SharedCoordinates& tc) const
		{
			assert((posAB == 0) || (posAB == 1));
			assert((colorIndex >= 0) && (colorIndex < m_maxVectorColor));

			return m_vectorColorBasis[colorIndex * m_numOfChannels + posAB];
		}
	};
}