#include <FoliageGeneratorBase.hpp>
#include <PlantSystem.hpp>

using namespace PlantFactory;
std::shared_ptr<Texture2D> DefaultFoliageGenerator::m_leafSurfaceTex = nullptr;

void DefaultFoliageGenerator::GenerateLeaves(
    Entity &internode, glm::mat4 &treeTransform,
    std::vector<glm::mat4> &leafTransforms, bool isLeft) {}

void DefaultFoliageGenerator::OnCreate() {
  m_defaultFoliageInfo = DefaultFoliageInfo();
  m_archetype = EntityManager::CreateEntityArchetype("Pine Foliage",
                                                     DefaultFoliageInfo());

  m_leafMaterial = AssetManager::CreateAsset<Material>();
  m_leafMaterial->SetProgram(
      DefaultResources::GLPrograms::StandardInstancedProgram);
  m_leafMaterial->m_alphaDiscardEnabled = true;
  m_leafMaterial->m_alphaDiscardOffset = 0.1f;
  m_leafMaterial->m_cullingMode = MaterialCullingMode::Off;
  if (!m_leafSurfaceTex)
    m_leafSurfaceTex = AssetManager::Import<Texture2D>(
        AssetManager::GetAssetFolderPath() / "Textures/Leaf/Pine/level0.png");
  //_LeafMaterial->SetTexture(_LeafSurfaceTex);
  m_leafMaterial->m_albedoColor =
      glm::normalize(glm::vec3(60.0f / 256.0f, 140.0f / 256.0f, 0.0f));
  m_leafMaterial->m_metallic = 0.0f;
  m_leafMaterial->m_roughness = 0.3f;
  m_leafMaterial->m_ambient = 1.0f;
}

void DefaultFoliageGenerator::Generate() {
  const auto tree = GetOwner();
  auto treeTransform = tree.GetDataComponent<GlobalTransform>();
  Entity foliageEntity;
  bool found = false;
  tree.ForEachChild([&found, &foliageEntity](Entity child) {
    if (child.HasDataComponent<DefaultFoliageInfo>()) {
      found = true;
      foliageEntity = child;
    }
  });
  if (!found) {
    foliageEntity = EntityManager::CreateEntity(m_archetype, "Foliage");
    foliageEntity.SetParent(tree);
    auto particleSys =
        foliageEntity.GetOrSetPrivateComponent<Particles>().lock();
    particleSys->m_material = m_leafMaterial;
    particleSys->m_mesh = DefaultResources::Primitives::Quad;
    particleSys->m_forwardRendering = false;
    Transform transform;
    transform.m_value =
        glm::translate(glm::vec3(0.0f)) * glm::scale(glm::vec3(1.0f));

    foliageEntity.SetDataComponent(transform);
    foliageEntity.SetDataComponent(m_defaultFoliageInfo);
  }
  auto particleSys = foliageEntity.GetOrSetPrivateComponent<Particles>().lock();
  particleSys->m_matrices->m_value.clear();
  GenerateLeaves(tree.GetChildren()[0], treeTransform.m_value,
                 particleSys->m_matrices->m_value, true);
}

void DefaultFoliageGenerator::OnGui() {
  if (ImGui::Button("Regenerate"))
    Generate();
  ImGui::DragFloat2("Leaf Size XY",
                    static_cast<float *>(
                        static_cast<void *>(&m_defaultFoliageInfo.m_leafSize)),
                    0.01f);
  ImGui::DragFloat("LeafIlluminationLimit",
                   &m_defaultFoliageInfo.m_leafIlluminationLimit, 0.01f);
  ImGui::DragFloat("LeafInhibitorFactor",
                   &m_defaultFoliageInfo.m_leafInhibitorFactor, 0.01f);
  ImGui::Checkbox("IsBothSide", &m_defaultFoliageInfo.m_isBothSide);
  ImGui::DragInt("SideLeafAmount", &m_defaultFoliageInfo.m_sideLeafAmount,
                 0.01f);
  ImGui::DragFloat("StartBendingAngle",
                   &m_defaultFoliageInfo.m_startBendingAngle, 0.01f);
  ImGui::DragFloat("BendingAngleIncrement",
                   &m_defaultFoliageInfo.m_bendingAngleIncrement, 0.01f);
  ImGui::DragFloat("LeafPhotoTropism", &m_defaultFoliageInfo.m_leafPhotoTropism,
                   0.01f);
  ImGui::DragFloat("LeafGravitropism", &m_defaultFoliageInfo.m_leafGravitropism,
                   0.01f);
  ImGui::DragFloat("LeafDistance", &m_defaultFoliageInfo.m_leafDistance, 0.01f);
}
void DefaultFoliageGenerator::Clone(
    const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<DefaultFoliageGenerator>(target);
}
