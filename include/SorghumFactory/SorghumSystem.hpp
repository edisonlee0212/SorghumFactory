#pragma once
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#endif
#include "SorghumField.hpp"
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include <Spline.hpp>
#include <sorghum_factory_export.h>
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API LeafTag : IDataComponent {
  int m_index = 0;
};
struct SORGHUM_FACTORY_API SorghumTag : IDataComponent {};
class SorghumProceduralDescriptor;
class SORGHUM_FACTORY_API SorghumSystem : public ISystem {
  static void ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
                              std::ofstream &of, unsigned &startIndex);
  bool m_ready = false;

public:
#pragma region Illumination
  int m_seed;
#ifdef RAYTRACERFACILITY
  OpenGLUtils::GLVBO m_lightProbeRenderingColorBuffer;
  OpenGLUtils::GLVBO m_lightProbeRenderingTransformBuffer;
  std::vector<glm::mat4> m_probeTransforms;
  std::vector<glm::vec4> m_probeColors;
  RayTracerFacility::IlluminationEstimationProperties m_properties;
  std::vector<Entity> m_processingEntities;
  int m_processingIndex;
  bool m_processing = false;
  float m_perPlantCalculationTime = 0.0f;
  bool m_displayLightProbes = true;
  void RenderLightProbes();
  void CalculateIllumination(
      const RayTracerFacility::IlluminationEstimationProperties &properties =
          RayTracerFacility::IlluminationEstimationProperties());
#endif
#pragma endregion

  EntityArchetype m_leafArchetype;
  EntityQuery m_leafQuery;
  EntityArchetype m_sorghumArchetype;
  EntityQuery m_sorghumQuery;

  const float m_leafNodeSphereSize = 0.1f;

  AssetRef m_leafNodeMaterial;
  AssetRef m_leafMaterial;

  AssetRef m_segmentedLeafMaterials[25];

  void OnCreate() override;
  void Start() override;
  Entity CreateSorghum(bool segmentedMask = false);
  Entity CreateSorghum(const std::shared_ptr<SorghumProceduralDescriptor>& descriptor, bool segmentedMask = false);
  Entity CreateSorghumLeaf(const Entity &plantEntity, int leafIndex);
  void GenerateMeshForAllSorghums(int segmentAmount = 2, int step = 2);
  Entity ImportPlant(const std::filesystem::path &path, const std::string &name,
                     bool segmentedMask = false);
  void OnInspect() override;
  void Update() override;
  void CreateGrid(RectangularSorghumFieldPattern &field,
                  const std::vector<Entity> &candidates);
  void CloneSorghums(const Entity &parent, const Entity &original,
                     std::vector<glm::mat4> &matrices);
  static void ExportSorghum(const Entity &sorghum, std::ofstream &of,
                            unsigned &startIndex);
  void ExportAllSorghumsModel(const std::string &filename);

  static void CollectEntities(std::vector<Entity> &entities,
                              const Entity &walker);

  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
};
} // namespace SorghumFactory
