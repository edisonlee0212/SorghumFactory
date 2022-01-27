#pragma once
#include <AutoSorghumGenerationPipeline.hpp>

#include <SorghumLayer.hpp>
#ifdef RAYTRACERFACILITY
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
#endif
using namespace SorghumFactory;
namespace Scripts {
enum class MultipleAngleCaptureStatus { Info, Mask, Angles };

struct CameraMatricesCollection{
  GlobalTransform m_camera;
  glm::mat4 m_projection;
  glm::mat4 m_view;
  std::string m_postFix;
};
struct SorghumInfo{
  GlobalTransform m_sorghum;
  std::string m_name;
};
class GeneralDataCapture : public IAutoSorghumGenerationPipelineBehaviour {
  MultipleAngleCaptureStatus m_captureStatus = MultipleAngleCaptureStatus::Info;
  bool SetUpCamera(AutoSorghumGenerationPipeline &pipeline);
  void ExportMatrices(const std::filesystem::path& path);
  std::vector<CameraMatricesCollection> m_cameraMatrices;
  std::vector<SorghumInfo> m_sorghumInfos;
  void CalculateMatrices();
  void Instantiate();
  Entity m_rayTracerCamera;

public:
  RayProperties m_rayProperties = {4, 512};
  AssetRef m_parameters;
  bool m_captureImage = true;
  bool m_captureMask = true;
  bool m_captureMesh = false;
  std::filesystem::path m_currentExportFolder = "Datasets/";
  int m_pitchAngleStart = 0;
  int m_pitchAngleStep = 20;
  int m_pitchAngleEnd = 40;
  int m_turnAngleStart = 0;
  int m_turnAngleStep = 120;
  int m_turnAngleEnd = 360;
  float m_fov = 60;
  float m_denoiserStrength = 1.0f;
  glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
  bool m_useClearColor = true;
  glm::vec3 m_backgroundColor = glm::vec3(0.2f);
  float m_cameraMin = 1;
  float m_cameraMax = 30;


  bool IsReady() override;
  void Start(AutoSorghumGenerationPipeline &pipeline) override;
  void End(AutoSorghumGenerationPipeline &pipeline) override;

  void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnInspect() override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace Scripts