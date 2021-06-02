#pragma once
#include <SorghumParameters.hpp>
using namespace UniEngine;
namespace PlantFactory {
	class SorghumData :
		public PrivateComponentBase
	{
	public:
		bool m_growthComplete = false;
		glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
		bool m_meshGenerated = false;
		SorghumParameters m_parameters;
		SorghumData();
		~SorghumData();
		void OnGui() override;
		void ExportModel(const std::string& filename, const bool& includeFoliage = true) const;
	};
}
