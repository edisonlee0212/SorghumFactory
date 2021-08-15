#pragma once
#include <TreeSystem.hpp>
using namespace UniEngine;
namespace PlantFactory {
class TreeLeaves final : public IPrivateComponent {
public:
  std::vector<int> m_targetBoneIndices;
  std::vector<glm::mat4> m_transforms;

  std::shared_ptr<Mesh> m_leavesMesh;
  std::shared_ptr<SkinnedMesh> m_skinnedLeavesMesh;

  void OnGui() override;
  void FormSkinnedMesh(std::vector<unsigned> &boneIndices);
  void FormMesh();
  void OnCreate() override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace PlantFactory
