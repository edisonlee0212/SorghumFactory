#pragma once
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#endif
#include "SorghumField.hpp"
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include <Spline.hpp>
#include "ILayer.hpp"
#include <sorghum_factory_export.h>
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API LeafTag : IDataComponent {
  int m_index = 0;
};
struct SORGHUM_FACTORY_API SorghumTag : IDataComponent {};
class SorghumProceduralDescriptor;
class SORGHUM_FACTORY_API SorghumLayer : public ILayer {
  static void ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
                              std::ofstream &of, unsigned &startIndex);

public:
#pragma region Illumination
  int m_seed = 0;
  float m_pushDistance = 0.001f;
  int m_sampleAmount = 300;
#ifdef RAYTRACERFACILITY
  bool m_enableMLVQ = false;
  int m_MLVQMaterialIndex = 1;
  std::vector<glm::mat4> m_probeTransforms;
  std::vector<glm::vec4> m_probeColors;
  std::vector<Entity> m_processingEntities;
  int m_processingIndex;
  bool m_processing = false;
  float m_lightProbeSize = 0.05f;
  float m_perPlantCalculationTime = 0.0f;
  bool m_displayLightProbes = true;

  void RenderLightProbes();
  void CalculateIlluminationFrameByFrame();
  void CalculateIllumination();
#endif
#pragma endregion
  EntityArchetype m_leafArchetype;
  EntityQuery m_leafQuery;
  EntityArchetype m_sorghumArchetype;
  EntityQuery m_sorghumQuery;

  AssetRef m_leafNodeMaterial;
  AssetRef m_leafMaterial;
  AssetRef m_leafAlbedoTexture;
  AssetRef m_leafNormalTexture;
  AssetRef m_segmentedLeafMaterials[25];

  void OnCreate() override;
  Entity CreateSorghum();
  Entity CreateSorghum(const std::shared_ptr<SorghumProceduralDescriptor>& descriptor);
  Entity CreateSorghumLeaf(const Entity &plantEntity, int leafIndex);
  void GenerateMeshForAllSorghums(int segmentAmount = 2, int step = 2);
  Entity ImportPlant(const std::filesystem::path &path, const std::string &name);
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
};
} // namespace SorghumFactory
