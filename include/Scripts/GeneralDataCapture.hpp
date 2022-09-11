#pragma once
#include <AutoSorghumGenerationPipeline.hpp>
#ifdef RAYTRACERFACILITY
#include <SorghumLayer.hpp>

#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;

using namespace PlantArchitect;
namespace Scripts {
struct SorghumInfo{
  GlobalTransform m_sorghum;
  std::string m_name;
};
class GeneralDataCapture : public IAutoSorghumGenerationPipelineBehaviour {
  void SetUpCamera(AutoSorghumGenerationPipeline &pipeline);
  std::vector<SorghumInfo> m_sorghumInfos;
  void Instantiate();
  Entity m_lab;
  Entity m_dirt;


  std::vector<glm::mat4> m_cameraModels;
  std::vector<glm::mat4> m_treeModels;
  std::vector<glm::mat4> m_projections;
  std::vector<glm::mat4> m_views;
  std::vector<std::string> m_names;
  void ExportMatrices(const std::filesystem::path& path);
public:
  RayProperties m_rayProperties = {6, 256};
  AssetRef m_labPrefab;
  AssetRef m_dirtPrefab;
  bool m_captureImage = true;
  bool m_captureMask = true;
  bool m_captureMesh = false;
  bool m_captureDepth = true;
  bool m_exportMatrices = true;
  std::filesystem::path m_currentExportFolder;
  int m_turnAngleStart = 0;
  int m_turnAngleStep = 1;
  int m_turnAngleEnd = 2;

  int m_topTurnAngleStart = 0;
  int m_topTurnAngleStep = 1;
  int m_topTurnAngleEnd = 2;

  float m_gamma = 2.2f;
  float m_fov = 30;
  float m_distanceToCenter = 8.2;
  float m_height = 0.66f;
  float m_topDistanceToCenter = 8.2;

  float m_denoiserStrength = 1.0f;
  glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
  bool m_useClearColor = true;
  glm::vec3 m_backgroundColor = glm::vec3(1.0f);
  float m_backgroundColorIntensity = 1.0f;
  float m_cameraMax = 10;


  void OnStart(AutoSorghumGenerationPipeline &pipeline) override;
  void OnEnd(AutoSorghumGenerationPipeline &pipeline) override;

  void OnBeforeGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnAfterGrowth(AutoSorghumGenerationPipeline &pipeline) override;
  void OnInspect() override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace Scripts
#endif