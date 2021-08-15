#pragma once
#include <TreeParameters.hpp>
using namespace UniEngine;
namespace PlantFactory {
class TreeData : public IPrivateComponent {
public:
#pragma region Info
  TreeParameters m_parameters;
  int m_currentSeed;
  float m_height;
  int m_maxBranchingDepth;
  int m_lateralBudsCount;
  float m_totalLength = 0;
#pragma endregion
#pragma region Runtime Data
  float m_activeLength = 0.0f;
  std::shared_ptr<Mesh> m_convexHull;


  std::shared_ptr<Mesh> m_branchMesh;

  std::shared_ptr<SkinnedMesh> m_skinnedBranchMesh;



  bool m_meshGenerated = false;
  bool m_foliageGenerated = false;
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
#pragma endregion
  void OnGui() override;
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;

  void OnCreate() override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace PlantFactory
