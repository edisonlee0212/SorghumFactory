#pragma once
#include <PlantManager.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
using namespace UniEngine;
namespace PlantFactory {
enum class GrowthParameterType {
  Gravitropism,
  Phototropism,
  InhibitorTransmitFactor,
  BranchingAngle,
  RollAngle,
  ApicalAngle,
  LateralBudPerNode,
  ResourceWeightApical,
  ResourceWeightLateral,
  AvoidanceAngle
};

struct TreeLeavesTag : IDataComponent {};

struct RbvTag : IDataComponent {};

class TreeManager {
protected:
#pragma region Class related
  TreeManager() = default;
  TreeManager(TreeManager &&) = default;
  TreeManager(const TreeManager &) = default;
  TreeManager &operator=(TreeManager &&) = default;
  TreeManager &operator=(const TreeManager &) = default;
#pragma endregion
#pragma region Helpers
  static void ExportChains(int parentOrder, Entity internode,
                           rapidxml::xml_node<> *chains,
                           rapidxml::xml_document<> *doc);
  static void WriteChain(int order, Entity internode,
                         rapidxml::xml_node<> *chains,
                         rapidxml::xml_document<> *doc);
  static Entity GetRootInternode(const Entity &tree);
  static Entity GetLeaves(const Entity &tree);
  static Entity GetRbv(const Entity &tree);
#pragma endregion
public:
#pragma region Physics
  float m_density = 1.0f;
  float m_linearDamping = 1.0f;
  float m_angularDamping = 1.0f;
  int m_positionSolverIteration = 8;
  int m_velocitySolverIteration = 8;
  float m_jointDriveStiffnessFactor = 500.0f;
  float m_jointDriveStiffnessThicknessFactor = 4.0f;
  float m_jointDriveDampingFactor = 1.0f;
  float m_jointDriveDampingThicknessFactor = 1.0f;
#pragma endregion

#pragma region Rendering
  VoxelSpace m_voxelSpaceModule;

  /**
   * \brief The information displayed when rendering internodes.
   */
  BranchRenderType m_branchRenderType = BranchRenderType::Order;
  PointerRenderType m_pointerRenderType = PointerRenderType::Illumination;
  float m_displayTime = 0;
  float m_previousGlobalTime = 0;
  float m_connectionWidth = 0.03f;
  bool m_displayThickness = true;

  bool m_updateBranch = false;
  bool m_updatePointer = false;
  bool m_alwaysUpdate = true;
  float m_pointerLength = 0.4f;
  float m_pointerWidth = 0.02f;

  bool m_drawVoxels = false;
  bool m_drawBranches = true;
  bool m_drawPointers = false;

  std::vector<glm::vec3> m_randomColors;
  float m_transparency = 0.7f;
  bool m_useTransparency = true;
  bool m_useColorMap = true;
  bool m_enableBranchDataCompress = false;
  float m_branchCompressFactor = 1.0f;

  bool m_enablePointerDataCompress = false;
  float m_pointerCompressFactor = 1.0f;

  int m_colorMapSegmentAmount = 3;
  std::vector<float> m_colorMapValues;
  std::vector<glm::vec3> m_colorMapColors;

  glm::vec4 m_pointerColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);

  std::vector<Entity> m_entitiesWithRenderer;
  OpenGLUtils::GLVBO m_internodeColorBuffer;
#pragma endregion
#pragma region Crown shyness
  float m_crownShynessDiameter = 0.2f;
#pragma endregion

#pragma region Leaf Gen
  int m_leafAmount = 20;
  float m_radius = 0.3f;
  glm::vec2 m_leafSize = glm::vec2(0.05f);
#pragma endregion

#pragma region Internode debugging camera
  CameraComponent m_internodeDebuggingCamera;
  int m_internodeDebuggingCameraResolutionX = 1;
  int m_internodeDebuggingCameraResolutionY = 1;
  float m_lastX = 0;
  float m_lastY = 0;
  float m_lastScrollY = 0;
  bool m_startMouse = false;
  bool m_startScroll = false;
  bool m_rightMouseButtonHold = false;

  Entity m_currentFocusingInternode = Entity();

#pragma endregion
  float m_meshResolution = 0.02f;
  float m_meshSubdivision = 4.0f;

  EntityArchetype m_leavesArchetype;
  EntityArchetype m_rbvArchetype;
  std::shared_ptr<Texture2D> m_defaultRayTracingBranchAlbedoTexture;
  std::shared_ptr<Texture2D> m_defaultRayTracingBranchNormalTexture;
  std::shared_ptr<Texture2D> m_defaultBranchAlbedoTexture;
  std::shared_ptr<Texture2D> m_defaultBranchNormalTexture;

  static void Update();
  static Entity CreateTree(const Transform &transform);
  static void OnGui();

  static void UpdateBranchCylinder(const bool &displayThickness,
                                   const float &width = 0.01f);
  static void UpdateBranchPointer(const float &length,
                                  const float &width = 0.01f);
  static void UpdateBranchColors();
  static void ColorSet(glm::vec4 &target, const float &value);

  static void RenderBranchCylinders(const float &displayTime);
  static void RenderBranchPointers(const float &displayTime);
  static void TreeNodeWalker(std::vector<Entity> &boundEntities,
                             std::vector<int> &parentIndices,
                             const int &parentIndex, const Entity &node);
  static TreeManager &GetInstance();
  static void TreeMeshGenerator(std::vector<Entity> &internodes,
                                std::vector<int> &parentIndices,
                                std::vector<Vertex> &vertices,
                                std::vector<unsigned> &indices);
  static void TreeSkinnedMeshGenerator(std::vector<Entity> &internodes,
                                       std::vector<int> &parentIndices,
                                       std::vector<SkinnedVertex> &vertices,
                                       std::vector<unsigned> &indices);
  static void GenerateMeshForTree(PlantManager &manager);
  static void FormCandidates(PlantManager &manager,
                             std::vector<InternodeCandidate> &candidates);
  static float GetGrowthParameter(const GrowthParameterType &type,
                                  TreeData &treeData,
                                  InternodeInfo &internodeInfo,
                                  InternodeGrowth &internodeGrowth,
                                  InternodeStatistics &internodeStatistics);
  static void PruneTrees(PlantManager &manager,
                         std::vector<Volume *> &obstacles);
  static void UpdateTreesMetaData(PlantManager &manager);
#pragma region Metadata
  static void UpdateDistances(const Entity &internode, TreeData &treeData);
  static void UpdateLevels(const Entity &internode, TreeData &treeData);
#pragma endregion
  static void ResetTimeForTree(const float &value);
  static void ResetTimeForTree(const Entity &internode,
                               const float &globalTime);
  static void
  DistributeResourcesForTree(PlantManager &manager,
                             std::vector<ResourceParcel> &totalNutrients);
  static void Init();
  static void SerializeScene(const std::string &filename);
  static void Serialize(const Entity &treeEntity, rapidxml::xml_document<> &doc,
                        rapidxml::xml_node<> *sceneNode);
};
} // namespace PlantFactory