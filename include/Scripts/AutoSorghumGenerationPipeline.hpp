#pragma once
#include <SorghumLayer.hpp>

using namespace SorghumFactory;

namespace Scripts {
enum class AutoSorghumGenerationPipelineStatus {
  Idle,
  BeforeGrowth,
  Growth,
  AfterGrowth
};
class AutoSorghumGenerationPipeline : public IPrivateComponent {
  int m_remainingInstanceAmount = 0;
public:
  std::string m_prefix;
  int m_startIndex = 1;
  int m_generationAmount = 1;
  Entity m_currentGrowingSorghum;
  AutoSorghumGenerationPipelineStatus m_status =
      AutoSorghumGenerationPipelineStatus::Idle;
  AssetRef m_pipelineBehaviour;
  bool m_busy = false;
  AssetRef m_currentUsingDescriptor;
  std::vector<AssetRef> m_descriptors;
int GetSeed() const;
  void Update() override;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};

class IAutoSorghumGenerationPipelineBehaviour : public IAsset {
public:
  virtual void OnStart(AutoSorghumGenerationPipeline &pipeline) = 0;
  virtual void OnEnd(AutoSorghumGenerationPipeline &pipeline) = 0;
  virtual void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline);
  virtual void OnGrowth(AutoSorghumGenerationPipeline &pipeline);
  virtual void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline);
};
} // namespace Scripts