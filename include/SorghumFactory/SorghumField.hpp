#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API SorghumFieldPattern {
public:
  virtual void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList){};
};

class SORGHUM_FACTORY_API RectangularSorghumFieldPattern : public SorghumFieldPattern {
public:
  glm::ivec2 m_size = glm::ivec2(4, 4);
  glm::vec2 m_distances = glm::vec2(2, 2);
  glm::vec3 m_rotationVariation = glm::vec3(0, 0, 0);
  void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList) override;
};


class SORGHUM_FACTORY_API SorghumField : public IAsset {
  friend class SorghumSystem;
  std::vector<AssetRef> m_newSorghumParameters;
  std::vector<glm::vec3> m_newSorghumPositions;
  std::vector<glm::vec3> m_newSorghumRotations;
  int m_newSorghumAmount = 1;
public:
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
}