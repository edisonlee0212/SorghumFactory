#pragma once
#include <AutoSorghumGenerationPipeline.hpp>

#include <SorghumLayer.hpp>
using namespace SorghumFactory;
namespace Scripts {
class PointCloudCapture : public IAutoSorghumGenerationPipelineBehaviour {
  Entity m_currentSorghumField;
  Entity m_ground;
  void Instantiate();
public:
  PointCloudSampleSettings m_settings;
  void Reset(Scripts::AutoSorghumGenerationPipeline &pipeline);
  AssetRef m_positionsField;
  std::filesystem::path m_currentExportFolder = "Datasets/";
  void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnInspect() override;
  void Start(AutoSorghumGenerationPipeline &pipeline) override;
  void End(AutoSorghumGenerationPipeline &pipeline) override;
  bool IsReady() override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
}