#pragma once
#include <sorghum_factory_export.h>

#include <SorghumParameters.hpp>
using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API SorghumData : public IPrivateComponent {
public:
  bool m_forceSameRotation = false;
  bool m_growthComplete = false;
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;

  int m_segmentAmount = 2;
  int m_step = 2;

  SorghumParameters m_parameters;
  void OnCreate() override;
  void OnDestroy() override;
  void OnGui() override;
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;

  void ApplyParameters();
};
} // namespace PlantFactory
