#include <TreeLeaves.hpp>

using namespace PlantFactory;

void TreeLeaves::OnGui() {
    ImGui::Text("Amount: %d", m_transforms.size());
}
void TreeLeaves::FormSkinnedMesh(std::vector<unsigned> &boneIndices) {
  auto quadMesh = DefaultResources::Primitives::Quad;
  auto &quadTriangles = quadMesh->UnsafeGetTriangles();
  auto quadVerticesSize = quadMesh->GetVerticesAmount();
  size_t offset = 0;
  size_t vi = 0;
  size_t ii = 0;
  std::vector<SkinnedVertex> skinnedVertices;
  std::vector<glm::uvec3> skinnedTriangles;

  skinnedVertices.resize(m_transforms.size() * quadVerticesSize);
  skinnedTriangles.resize(m_transforms.size() * quadTriangles.size());
  int mi = 0;
  for (const auto &matrix : m_transforms) {
    auto boneIndex = m_targetBoneIndices[mi];
    for (auto i = 0; i < quadMesh->GetVerticesAmount(); i++) {
      skinnedVertices[vi].m_position = matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_position, 1.0f);
      skinnedVertices[vi].m_normal = glm::normalize(
          glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
      skinnedVertices[vi].m_tangent = glm::normalize(
          glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
      skinnedVertices[vi].m_texCoords = quadMesh->UnsafeGetVertices()[i].m_texCoords;
      skinnedVertices[vi].m_bondId = glm::ivec4(boneIndex, -1, -1, -1);
      skinnedVertices[vi].m_weight = glm::vec4(1, 0, 0, 0);
      skinnedVertices[vi].m_bondId2 = glm::ivec4(-1, -1, -1, -1);
      skinnedVertices[vi].m_weight2 = glm::vec4(0, 0, 0, 0);
      vi++;
    }
    for (auto triangle : quadTriangles) {
      triangle.x += offset;
      triangle.y += offset;
      triangle.z += offset;
      skinnedTriangles[ii] = triangle;
      ii++;
    }
    offset += quadVerticesSize;
    mi++;
  }
  auto skinnedMeshRenderer = GetOwner().GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
  m_skinnedLeavesMesh->SetVertices(17, skinnedVertices, skinnedTriangles);
  m_skinnedLeavesMesh->m_boneAnimatorIndices = boneIndices;

  skinnedMeshRenderer->m_skinnedMesh.Set<SkinnedMesh>(m_skinnedLeavesMesh);

  skinnedMeshRenderer->SetEnabled(true);
  if(GetOwner().HasPrivateComponent<MeshRenderer>()){
    GetOwner().GetOrSetPrivateComponent<MeshRenderer>().lock()->SetEnabled(false);
  }
}

void TreeLeaves::FormMesh() {
    auto quadMesh = DefaultResources::Primitives::Quad;
    auto &quadTriangles = quadMesh->UnsafeGetTriangles();
    auto quadVerticesSize = quadMesh->GetVerticesAmount();
    size_t offset = 0;
    size_t vi = 0;
    size_t ii = 0;

    std::vector<Vertex> vertices;
    std::vector<glm::uvec3> triangles;

    vertices.resize(m_transforms.size() * quadVerticesSize);
    triangles.resize(m_transforms.size() * quadTriangles.size());
    for (const auto &matrix : m_transforms) {
        for (auto i = 0; i < quadMesh->GetVerticesAmount(); i++) {
            vertices[vi].m_position = matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_position, 1.0f);
            vertices[vi].m_normal = glm::normalize(
                    glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_normal, 0.0f)));
            vertices[vi].m_tangent = glm::normalize(
                    glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertices()[i].m_tangent, 0.0f)));
            vertices[vi].m_texCoords = quadMesh->UnsafeGetVertices()[i].m_texCoords;
            vi++;
        }
        for (auto triangle : quadTriangles) {
            triangle.x += offset;
            triangle.y += offset;
            triangle.z += offset;
            triangles[ii] = triangle;
            ii++;
        }
        offset += quadVerticesSize;
    }
    m_leavesMesh->SetVertices(17, vertices, triangles);
    auto meshRenderer = GetOwner().GetOrSetPrivateComponent<MeshRenderer>().lock();
    meshRenderer->m_mesh.Set<Mesh>(m_leavesMesh);

    meshRenderer->SetEnabled(true);
    if(GetOwner().HasPrivateComponent<SkinnedMeshRenderer>()){
      GetOwner().GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock()->SetEnabled(false);
    }
}
void TreeLeaves::Clone(const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<TreeLeaves>(target);


}
void TreeLeaves::OnCreate() {
  m_leavesMesh = AssetManager::CreateAsset<Mesh>();
  m_skinnedLeavesMesh = AssetManager::CreateAsset<SkinnedMesh>();
}
