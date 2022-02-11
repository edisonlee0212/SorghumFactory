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

struct SorghumInfo{
  GlobalTransform m_sorghum;
  std::string m_name;
};
class GeneralDataCapture : public IAutoSorghumGenerationPipelineBehaviour {
  MultipleAngleCaptureStatus m_captureStatus = MultipleAngleCaptureStatus::Info;
  bool SetUpCamera(AutoSorghumGenerationPipeline &pipeline);
  std::vector<SorghumInfo> m_sorghumInfos;
  void Instantiate();
  Entity m_rayTracerCamera;
  Entity m_lab;
  Entity m_dirt;
public:
  RayProperties m_rayProperties = {6, 256};
  AssetRef m_parameters;
  AssetRef m_labPrefab;
  AssetRef m_dirtPrefab;
  bool m_captureImage = true;
  bool m_captureMask = true;
  bool m_captureMesh = false;
  std::filesystem::path m_currentExportFolder = "Datasets";
  int m_turnAngleStart = 0;
  int m_turnAngleStep = 72;
  int m_turnAngleEnd = 288;
  float m_gamma = 2.2f;
  float m_fov = 30;
  float m_distanceToCenter = 8.2;
  float m_height = 0.66f;

  float m_denoiserStrength = 1.0f;
  glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
  bool m_useClearColor = true;
  glm::vec3 m_backgroundColor = glm::vec3(1.0f);
  float m_backgroundColorIntensity = 1.25f;
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