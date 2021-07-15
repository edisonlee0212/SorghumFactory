#pragma once
#include <TreeManager.hpp>
using namespace UniEngine;
namespace PlantFactory
{
	class TreeLeaves final : public IPrivateComponent
	{
	public:
	    std::vector<int> m_targetBoneIndices;
		std::vector<glm::mat4> m_transforms;
		void OnGui() override;
		void FormMesh(std::vector<unsigned>& boneIndices);
	};
}


