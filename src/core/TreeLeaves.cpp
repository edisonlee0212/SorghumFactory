#include <TreeLeaves.hpp>

using namespace PlantFactory;

void TreeLeaves::OnGui()
{
	ImGui::Text("Amount: %d", m_transforms.size());
}

void TreeLeaves::FormMesh()
{
	auto quadMesh = DefaultResources::Primitives::Quad;
	auto& quadTriangles = quadMesh->UnsafeGetTriangles();
	auto quadVerticesSize = quadMesh->GetVerticesAmount();
	size_t offset = 0;
	std::vector<Vertex> vertices;
	std::vector<glm::uvec3> triangles;
	bool fromNew = true;

	vertices.resize(m_transforms.size() * quadVerticesSize);
	triangles.resize(m_transforms.size() * quadTriangles.size());
	size_t vi = 0;
	size_t ii = 0;
	for (auto& matrix : m_transforms)
	{
		for (auto i = 0; i < quadMesh->GetVerticesAmount(); i++) {
			vertices[vi].m_position = matrix * glm::vec4(quadMesh->UnsafeGetVertexPositions()[i], 1.0f);
			vertices[vi].m_normal = glm::normalize(glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertexNormals()[i], 0.0f)));
			vertices[vi].m_tangent = glm::normalize(glm::vec3(matrix * glm::vec4(quadMesh->UnsafeGetVertexTangents()[i], 0.0f)));
			vertices[vi].m_texCoords = quadMesh->UnsafeGetVertexTexCoords()[i];
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
		offset += quadVerticesSize;
	}
	auto& meshRenderer = GetOwner().GetPrivateComponent<MeshRenderer>();
	auto& mesh = meshRenderer->m_mesh;
	mesh->SetVertices(17, vertices, triangles);
}
