#pragma once
#include <sorghum_factory_export.h>
#include <CUDAModule.hpp>
#include <Curve.hpp>
#include <LeafSegment.hpp>

#include <Spline.hpp>
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API LeafTag : IDataComponent {};
struct SORGHUM_FACTORY_API SorghumTag : IDataComponent {};



class SORGHUM_FACTORY_API SorghumField {
public:
  virtual void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList){};
};

class SORGHUM_FACTORY_API RectangularSorghumField : public SorghumField {
public:
  glm::ivec2 m_size = glm::ivec2(4, 4);
  glm::vec2 m_distances = glm::vec2(2, 2);
  void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList) override;
};

class SORGHUM_FACTORY_API SorghumSystem : public ISystem {
  static void ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
                              std::ofstream &of, unsigned &startIndex);

  bool m_ready = false;
public:
#pragma region Illumination
  int m_seed;
  OpenGLUtils::GLVBO m_lightProbeRenderingColorBuffer;
  OpenGLUtils::GLVBO m_lightProbeRenderingTransformBuffer;
  std::vector<glm::mat4> m_probeTransforms;
  std::vector<glm::vec4> m_probeColors;

  RayTracerFacility::IlluminationEstimationProperties m_properties;
  std::vector<Entity> m_processingEntities;
  int m_processingIndex;
  bool m_processing = false;
  float m_perPlantCalculationTime = 0.0f;
#pragma endregion
  bool m_displayLightProbes = true;
  EntityArchetype m_leafArchetype;
  EntityQuery m_leafQuery;
  EntityArchetype m_sorghumArchetype;
  EntityQuery m_sorghumQuery;

  const float m_leafNodeSphereSize = 0.1f;

  AssetRef m_leafSurfaceTexture;
  AssetRef m_leafNormalTexture;

  AssetRef m_rayTracedLeafSurfaceTexture;
  AssetRef m_rayTracedLeafNormalTexture;

  AssetRef m_leafNodeMaterial;
  AssetRef m_leafMaterial;
  AssetRef m_instancedLeafMaterial;

  void OnCreate() override;
  void Start() override;
  Entity CreateSorghum();
  Entity CreateSorghumLeaf(const Entity &plantEntity);
  void GenerateMeshForAllSorghums(int segmentAmount = 2, int step = 2);
  Entity ImportPlant(const std::filesystem::path &path, const std::string &name);
  void OnInspect() override;
  void Update() override;
  void CreateGrid(SorghumField &field,
                         const std::vector<Entity> &candidates);
  void CloneSorghums(const Entity &parent, const Entity &original,
                            std::vector<glm::mat4> &matrices);
  static void ExportSorghum(const Entity &sorghum, std::ofstream &of,
                            unsigned &startIndex);
  void ExportAllSorghumsModel(const std::string &filename);
  void RenderLightProbes();
  static void CollectEntities(std::vector<Entity> &entities,
                              const Entity &walker);
  void CalculateIllumination(
      const RayTracerFacility::IlluminationEstimationProperties &properties =
          RayTracerFacility::IlluminationEstimationProperties());

  void Relink(const std::unordered_map<Handle, Handle> &map) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;

  void CollectAssetRef(std::vector<AssetRef> &list) override;
};
} // namespace PlantFactory
