#include <FileUtil.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
using namespace RayTracerFacility;
std::string FileIO::LoadFileAsString(const std::string& path)
{
	std::ifstream file;
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		file.open(path);
		std::stringstream stream;
		// read file's buffer contents into streams
		stream << file.rdbuf();
		// close file handlers
		file.close();
		// convert stream into string
		return stream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cerr << "Failed to load file." << std::endl;
		return "";
	}
}
