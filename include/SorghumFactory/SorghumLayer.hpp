#pragma once
#ifdef RAYTRACERFACILITY
#include <CUDAModule.hpp>
#endif
#include "ILayer.hpp"
#include "PointCloud.hpp"
#include "SorghumField.hpp"
#include <ICurve.hpp>
#include <LeafSegment.hpp>
#include <Spline.hpp>
#include <sorghum_factory_export.h>
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API LeafTag : IDataComponent {};
struct SORGHUM_FACTORY_API PanicleTag : IDataComponent {};
struct SORGHUM_FACTORY_API StemTag : IDataComponent {};
struct SORGHUM_FACTORY_API SorghumTag : IDataComponent {};
class SorghumStateGenerator;



class SORGHUM_FACTORY_API SorghumLayer : public ILayer {
  static void ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
                              std::ofstream &of, unsigned &startIndex);

public:
#ifdef RAYTRACERFACILITY
#pragma region Illumination
  int m_seed = 0;
  float m_pushDistance = 0.001f;
  RayTracerFacility::RayProperties m_rayProperties;

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

#pragma endregion
#endif
  bool m_autoRefreshSorghums = true;
  EntityArchetype m_leafArchetype;
  EntityQuery m_leafQuery;
  EntityArchetype m_sorghumArchetype;
  EntityQuery m_sorghumQuery;
  EntityArchetype m_panicleArchetype;
  EntityQuery m_panicleQuery;
  EntityArchetype m_stemArchetype;
  EntityQuery m_stemQuery;

  AssetRef m_panicleMaterial;
  AssetRef m_leafMaterial;
  AssetRef m_leafAlbedoTexture;
  AssetRef m_leafNormalTexture;
  AssetRef m_segmentedLeafMaterials[25];

  float m_verticalSubdivisionMaxUnitLength = 0.01f;
  int m_horizontalSubdivisionStep = 4;
  float m_skeletonWidth = 0.0025f;

  glm::vec3 m_skeletonColor = glm::vec3(0);

  void OnCreate() override;
  Entity CreateSorghum();
  Entity CreateSorghum(const std::shared_ptr<ProceduralSorghum> &descriptor);
  Entity
  CreateSorghum(const std::shared_ptr<SorghumStateGenerator> &descriptor);
  Entity CreateSorghumStem(const Entity &plantEntity);
  Entity CreateSorghumLeaf(const Entity &plantEntity, int leafIndex);
  Entity CreateSorghumPanicle(const Entity &plantEntity);
  void GenerateMeshForAllSorghums();
  void OnInspect() override;
  void Update() override;
  void LateUpdate() override;

  static void ExportSorghum(const Entity &sorghum, std::ofstream &of,
                            unsigned &startIndex);
  void ExportAllSorghumsModel(const std::string &filename);


};

} // namespace SorghumFactory
