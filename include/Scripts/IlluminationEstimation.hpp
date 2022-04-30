#pragma once
#ifdef RAYTRACERFACILITY
#include "GeneralAutomatedPipeline.hpp"
#include "RayTracer.hpp"
#include <SorghumLayer.hpp>
using namespace SorghumFactory;
using namespace RayTracerFacility;
namespace Scripts {
class IlluminationEstimationPipeline
    : public IGeneralAutomatedPipelineBehaviour {
  bool m_running = false;
  float m_currentTime = 0;

public:
  void ExportCSV(const std::filesystem::path &path) const;
  void Clear();
  float m_timeInterval = 5;
  RayProperties m_rayProperties = {8, 1000};
  AssetRef m_skyIlluminance;
  std::vector<AssetRef> m_sensorGroups;
  std::vector<std::pair<float, std::vector<float>>> m_results;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void OnStart(GeneralAutomatedPipeline &pipeline);
  void OnEnd(GeneralAutomatedPipeline &pipeline);
  void OnBeforeProcessing(GeneralAutomatedPipeline &pipeline);
  void OnProcessing(GeneralAutomatedPipeline &pipeline);
  void OnAfterProcessing(GeneralAutomatedPipeline &pipeline);
};
} // namespace Scripts
#endif