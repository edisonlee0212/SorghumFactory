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
  friend class SorghumLayer;
  std::vector<AssetRef> m_newSorghumParameters;
  std::vector<glm::vec3> m_newSorghumPositions;
  std::vector<glm::vec3> m_newSorghumRotations;
  int m_newSorghumAmount = 1;
public:
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
};
template <typename T>
inline void SaveListAsBinary(const std::string& name, const std::vector<T>& target, YAML::Emitter &out){
  if (!target.empty())
  {
    out << YAML::Key << name << YAML::Value
        << YAML::Binary((const unsigned char *)target.data(), target.size() * sizeof(T));
  }
}
template <typename T>
inline void LoadListFromBinary(const std::string& name, std::vector<T>& target, const YAML::Node &in){
  if (in[name])
  {
    auto binaryList = in[name].as<YAML::Binary>();
    target.resize(binaryList.size() / sizeof(T));
    std::memcpy(target.data(), binaryList.data(), binaryList.size());
  }
}
}