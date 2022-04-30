#pragma once
#include <SorghumLayer.hpp>
using namespace SorghumFactory;
namespace Scripts {
enum class GeneralAutomatedPipelineStatus {
  Idle,
  BeforeProcessing,
  Processing,
  AfterProcessing
};
class GeneralAutomatedPipeline : public IPrivateComponent {
  int m_remainingTaskAmount = 0;

public:
  int m_startIndex = 1;
  int m_taskAmount = 1;
  GeneralAutomatedPipelineStatus m_status =
      GeneralAutomatedPipelineStatus::Idle;
  AssetRef m_pipelineBehaviour;
  bool m_busy = false;
  void OnDestroy() override;
  void Update() override;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};

class IGeneralAutomatedPipelineBehaviour : public IAsset {
public:
  virtual void OnStart(GeneralAutomatedPipeline &pipeline) = 0;
  virtual void OnEnd(GeneralAutomatedPipeline &pipeline) = 0;
  virtual void OnBeforeProcessing(GeneralAutomatedPipeline &pipeline) {};
  virtual void OnProcessing(GeneralAutomatedPipeline &pipeline) {};
  virtual void OnAfterProcessing(GeneralAutomatedPipeline &pipeline) {};
};
} // namespace Scripts