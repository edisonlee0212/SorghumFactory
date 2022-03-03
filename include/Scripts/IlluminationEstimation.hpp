#pragma once
#ifdef RAYTRACERFACILITY
#include <SorghumLayer.hpp>
#include "RayTracer.hpp"
using namespace SorghumFactory;
using namespace RayTracerFacility;
namespace Scripts {
class IlluminationEstimation : public IPrivateComponent {
  bool m_running = false;
  float m_currentTime = 0;
public:
  void ExportCSV(const std::filesystem::path &path, const std::vector<float>& results) const;
  void Clear();
  float m_timeInterval = 5;
  RayProperties m_rayProperties = {8, 1000};
  AssetRef m_skyIlluminance;
  AssetRef m_PARSensorGroup1;
  AssetRef m_PARSensorGroup2;
  AssetRef m_PARSensorGroup3;
  std::vector<float> m_PAR1Result;
  std::vector<float> m_PAR2Result;
  std::vector<float> m_PAR3Result;
  void OnInspect() override;
  void Update() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
}
#endif