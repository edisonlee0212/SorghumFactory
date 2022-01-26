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
  void DropBehaviourButton();
public:
  int m_currentIndex = 0;
  int m_startIndex = 0;
  int m_endIndex = 2;
  Entity m_currentGrowingSorghum;
  AutoSorghumGenerationPipelineStatus m_status =
      AutoSorghumGenerationPipelineStatus::Idle;
  AssetRef m_pipelineBehaviour;
  void Update() override;
  void OnInspect() override;
};

class IAutoSorghumGenerationPipelineBehaviour : public IAsset {
public:
  virtual bool IsReady() = 0;
  virtual void Start(AutoSorghumGenerationPipeline &pipeline) = 0;
  virtual void End(AutoSorghumGenerationPipeline &pipeline) = 0;
  virtual void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline);
  virtual void OnGrowth(AutoSorghumGenerationPipeline &pipeline);
  virtual void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline);
};
} // namespace Scripts