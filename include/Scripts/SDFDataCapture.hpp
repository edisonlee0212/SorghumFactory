#pragma once
#include <AutoSorghumGenerationPipeline.hpp>

#include <SorghumLayer.hpp>
using namespace SorghumFactory;
namespace Scripts {
enum class MultipleAngleCaptureStatus { Info, Mask, Angles };
class SDFDataCapture : public IAutoSorghumGenerationPipelineBehaviour {
  int m_pitchAngle = -1;
  int m_turnAngle = -1;
  Entity m_currentGrowingSorghum;
  int m_remainingInstanceAmount = 0;
  bool m_skipCurrentFrame = false;

  MultipleAngleCaptureStatus m_captureStatus = MultipleAngleCaptureStatus::Info;
  bool SetUpCamera();
  void ExportMatrices(const std::filesystem::path& path);
  glm::vec3 m_cameraPosition;
  glm::quat m_cameraRotation;
public:
  AssetRef m_parameters;
  bool m_captureImage = true;
  bool m_captureMask = true;
  bool m_captureDepth = true;
  bool m_captureMesh = true;
  std::filesystem::path m_currentExportFolder = "export/";

  glm::vec3 m_focusPoint = glm::vec3(0, 3, 0);
  float m_pitchAngleStart = 0;
  float m_pitchAngleStep = 20;
  float m_pitchAngleEnd = 60;
  float m_turnAngleStep = 120;
  float m_distance = 20;
  float m_fov = 60;
  glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
  EntityRef m_cameraEntity;
  int m_generationAmount = 5;

  bool m_useClearColor = true;
  glm::vec3 m_backgroundColor = glm::vec3(1.0f);
  float m_cameraMin = 1;
  float m_cameraMax = 30;
  std::vector<glm::mat4> m_cameraModels;
  std::vector<glm::mat4> m_sorghumModels;
  std::vector<glm::mat4> m_projections;
  std::vector<glm::mat4> m_views;
  std::vector<std::string> m_names;
  void OnIdle(AutoSorghumGenerationPipeline &pipeline) override;
  void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnInspect() override;
};
} // namespace Scripts