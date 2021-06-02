#pragma once
#include <TreeManager.hpp>
using namespace UniEngine;
namespace PlantFactory
{
	class TreeLeaves final : public PrivateComponentBase
	{
	public:
		std::vector<glm::mat4> m_transforms;
		void OnGui() override;
		void FormMesh();
	};
}


