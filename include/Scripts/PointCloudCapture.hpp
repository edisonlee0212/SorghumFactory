#pragma once
#include <AutoSorghumGenerationPipeline.hpp>

#include <SorghumLayer.hpp>
using namespace SorghumFactory;
namespace Scripts {
class PointCloudCapture : public IAutoSorghumGenerationPipelineBehaviour {
  int m_currentIndex = -1;
  Entity m_currentSorghum;
  Entity m_currentSorghumField;
public:
  void Reset();
  AssetRef m_positionsField;
  int m_startIndex = 0;
  int m_endIndex = 2;
  std::filesystem::path m_currentExportFolder = "export/";
  void OnIdle(AutoSorghumGenerationPipeline &pipeline) override;
  void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnInspect() override;
};
}