#pragma once
#include <TreeManager.hpp>
#include <concurrent_vector.h>
using namespace UniEngine;
namespace PlantFactory
{
	class TreeLeaves final : public PrivateComponentBase
	{
	public:
		Concurrency::concurrent_vector<glm::mat4> m_transforms;
		void OnGui() override;
		void FormMesh();
	};
}


