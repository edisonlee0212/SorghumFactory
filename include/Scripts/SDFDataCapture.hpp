#pragma once
#include <AutoSorghumGenerationPipeline.hpp>
#include <SorghumProceduralDescriptor.hpp>
#include <SorghumSystem.hpp>
using namespace SorghumFactory;
namespace Scripts {
class SDFDataCapture : public IAutoSorghumGenerationPipelineBehaviour {
  int m_pitchAngle = -1;
  int m_turnAngle = -1;
  Entity m_currentGrowingSorghum;
public:
  SorghumProceduralDescriptor m_parameters;

  bool m_segmentedMask = false;

  std::filesystem::path m_currentExportFolder = "./export/";

  glm::vec3 m_focusPoint = glm::vec3(0, 3, 0);
  float m_pitchAngleStart = 0;
  float m_pitchAngleStep = 10;
  float m_pitchAngleEnd = 30;
  float m_turnAngleStep = 90;
  float m_distance = 4.5;
  float m_fov = 60;
  glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
  EntityRef m_cameraEntity;


  bool m_useClearColor = true;
  glm::vec3 m_backgroundColor = glm::vec3(1.0f);
  float m_cameraMin = 1;
  float m_cameraMax = 30;
  std::vector<glm::mat4> m_cameraModels;
  std::vector<glm::mat4> m_sorghumModels;
  std::vector<glm::mat4> m_projections;
  std::vector<glm::mat4> m_views;
  std::vector<std::string> m_names;
  void OnIdle(AutoSorghumGenerationPipeline & pipeline) override;
  void OnBeforeGrowth(AutoSorghumGenerationPipeline & pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline & pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline & pipeline) override;
  void OnInspect() override;
};
}