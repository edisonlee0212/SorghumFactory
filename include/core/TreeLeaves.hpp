#pragma once
#include <TreeSystem.hpp>
using namespace UniEngine;
namespace PlantFactory {
class TreeLeaves final : public IPrivateComponent {
public:
  std::vector<int> m_targetBoneIndices;
  std::vector<glm::mat4> m_transforms;
  void OnGui() override;
  void FormSkinnedMesh(std::vector<unsigned> &boneIndices);
  void FormMesh();

  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace PlantFactory
