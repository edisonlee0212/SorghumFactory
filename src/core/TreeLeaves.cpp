#include <TreeLeaves.hpp>

using namespace PlantFactory;

void TreeLeaves::OnGui()
{
	ImGui::Text("Amount: %d", m_transforms.size());
}

void TreeLeaves::FormMesh()
{
	auto quadMesh = Default::Primitives::Quad;
	auto& quadVertices = quadMesh->UnsafeGetVertices();
	auto& quadTriangles = quadMesh->UnsafeGetTriangles();
	size_t offset = 0;
	std::vector<Vertex> vertices;
	std::vector<glm::uvec3> triangles;
	bool fromNew = true;

	vertices.resize(m_transforms.size() * quadVertices.size());
	triangles.resize(m_transforms.size() * quadTriangles.size());
	size_t vi = 0;
	size_t ii = 0;
	for (auto& matrix : m_transforms)
	{
		for (const auto& vertex : quadVertices) {
			vertices[vi].m_position = matrix * glm::vec4(vertex.m_position, 1.0f);
			vertices[vi].m_normal = glm::normalize(glm::vec3(matrix * glm::vec4(vertex.m_normal, 0.0f)));
			vertices[vi].m_tangent = glm::normalize(glm::vec3(matrix * glm::vec4(vertex.m_tangent, 0.0f)));
			vertices[vi].m_texCoords0 = vertex.m_texCoords0;
			vi++;
		}
		for (auto triangle : quadTriangles)
		{
			triangle.x += offset;
			triangle.y += offset;
			triangle.z += offset;
			triangles[ii] = triangle;
			ii++;
		}
		offset += quadVertices.size();
	}
	auto& meshRenderer = GetOwner().GetPrivateComponent<MeshRenderer>();
	std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
	mesh->SetVertices(17, vertices, triangles, true);
	meshRenderer->m_mesh = mesh;
}
