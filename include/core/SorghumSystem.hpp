#pragma once
#include <CUDAModule.hpp>
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include <PlantSystem.hpp>
#include <RayTracedRenderer.hpp>
using namespace UniEngine;
namespace PlantFactory {
struct LeafInfo : IDataComponent {};

struct PlantNode {
  glm::vec3 m_position;
  float m_theta;
  float m_width;
  glm::vec3 m_axis;
  bool m_isLeaf;
  PlantNode(glm::vec3 position, float angle, float width, glm::vec3 axis,
            bool isLeaf);
};

class Spline : public IPrivateComponent {
public:
  glm::vec3 m_left;
  float m_startingPoint;

  std::vector<PlantNode> m_nodes;

  std::vector<LeafSegment> m_segments;
  std::vector<BezierCurve> m_curves;
  std::vector<Vertex> m_vertices;
  std::vector<unsigned> m_indices;
  void Import(std::ifstream &stream);
  glm::vec3 EvaluatePointFromCurve(float point);
  glm::vec3 EvaluateAxisFromCurve(float point);
  void OnGui() override;

  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};

class SorghumField {
public:
  virtual void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList){};
};

class RectangularSorghumField : public SorghumField {
public:
  glm::ivec2 m_size = glm::ivec2(4, 4);
  glm::vec2 m_distances = glm::vec2(2, 2);
  void
  GenerateField(std::vector<std::vector<glm::mat4>> &matricesList) override;
};

class SorghumSystem : public ISystem {
  std::shared_ptr<PlantSystem> m_plantSystem;
  static void ObjExportHelper(glm::vec3 position, std::shared_ptr<Mesh> mesh,
                              std::ofstream &of, unsigned &startIndex);

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

  const float m_leafNodeSphereSize = 0.1f;

  std::shared_ptr<Texture2D> m_leafSurfaceTexture;
  std::shared_ptr<Texture2D> m_leafNormalTexture;
  std::shared_ptr<Material> m_leafNodeMaterial;
  std::shared_ptr<Material> m_leafMaterial;
  std::shared_ptr<Material> m_instancedLeafMaterial;
  void OnCreate() override;
  Entity CreateSorghum();
  Entity CreateSorghumLeaf(const Entity &plantEntity);
  void GenerateMeshForAllSorghums(int segmentAmount = 2, int step = 2);
  Entity ImportPlant(const std::filesystem::path &path, const std::string &name);
  void OnGui() override;
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
  void GenerateLeavesForSorghum();
  void FormCandidates(std::vector<InternodeCandidate> &candidates);
  void FormLeafNodes();
  void RemoveInternodes(const Entity &sorghum);

  void DeleteAllPlantsHelper();
};
} // namespace PlantFactory
