#pragma once
#ifdef RAYTRACERFACILITY
#include "GeneralAutomatedPipeline.hpp"
#include "RayTracer.hpp"
#include <SorghumLayer.hpp>
using namespace EcoSysLab;
using namespace RayTracerFacility;
namespace Scripts {
class IlluminationEstimationPipeline
    : public IGeneralAutomatedPipelineBehaviour {
  float m_currentTime = 0;

public:
  std::filesystem::path m_currentExportFolder;
  Entity Instantiate() override;
  void ExportCSV(const std::filesystem::path &path);
  float m_timeInterval = 5;
  RayProperties m_rayProperties = {8, 1000};
  AssetRef m_skyIlluminance;
  std::vector<AssetRef> m_sensorGroups;
  std::vector<std::pair<float, std::vector<float>>> m_results;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void OnStart(GeneralAutomatedPipeline &pipeline) override;
  void OnBeforeProcessing(GeneralAutomatedPipeline &pipeline) override;
  void OnProcessing(GeneralAutomatedPipeline &pipeline) override;
  void OnAfterProcessing(GeneralAutomatedPipeline &pipeline) override;
};
} // namespace Scripts
#endif