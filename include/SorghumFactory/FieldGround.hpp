#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
class SORGHUM_FACTORY_API FieldGround : public IPrivateComponent{
public:
  glm::vec2 m_scale;
  glm::ivec2 m_size;
  float m_rowWidth;
  float m_alleyDepth;

  float m_noiseScale;
  float m_noiseIntensity;
  void OnCreate() override;
  void GenerateMesh(float overrideDepth = -1.0f);
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
}