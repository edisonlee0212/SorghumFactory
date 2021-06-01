#pragma once
#include <raymlvq_export.h>
#include <sstream>
namespace Cuda {
	class RAYMLVQ_API FileIO
	{
	public:
		static std::string LoadFileAsString(const std::string& path = "");
	};
}
