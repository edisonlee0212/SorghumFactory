#pragma once
#include <sorghum_factory_export.h>

using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API SkyIlluminanceSnapshot {
  float m_ghi = 1000;
  float m_azimuth = 0;
  float m_zenith = 0;
  [[nodiscard]] glm::vec3 GetSunDirection();
  [[nodiscard]] float GetSunIntensity();
};
class SORGHUM_FACTORY_API SkyIlluminance : public IAsset {
public:
  std::map<float, SkyIlluminanceSnapshot> m_snapshots;
  float m_minTime;
  float m_maxTime;
  [[nodiscard]] SkyIlluminanceSnapshot Get(float time);
  void ImportCSV(const std::filesystem::path &path);
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace SorghumFactory