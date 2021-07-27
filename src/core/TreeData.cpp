#include <FoliageGeneratorBase.hpp>
#include <PlantSystem.hpp>
#include <TreeData.hpp>
#include <TreeSystem.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>

using namespace PlantFactory;


void TreeData::OnGui() {
    if (ImGui::TreeNodeEx("I/O")) {
        if (m_meshGenerated) {
            FileSystem::SaveFile("Export OBJ", ".obj", [this](const std::string &path) {
                                 ExportModel(path);
                             }
            );
        }
      FileSystem::SaveFile("Export xml graph", ".xml", [this](const std::string &path) {
                             std::ofstream ofs;
                             ofs.open(path.c_str(), std::ofstream::out | std::ofstream::trunc);
                             if (!ofs.is_open()) {
                                 Debug::Error("Can't open file!");
                                 return;
                             }
                             rapidxml::xml_document<> doc;
                             auto *type = doc.allocate_node(rapidxml::node_doctype, 0, "Tree");
                             doc.append_node(type);
                             auto *scene = doc.allocate_node(rapidxml::node_element, "Tree", "Tree");
                             doc.append_node(scene);
                             TreeSystem::Serialize(GetOwner(), doc, scene);
                             ofs << doc;
                             ofs.flush();
                             ofs.close();
                         }
        );
    }
    if (ImGui::TreeNodeEx("Runtime Data")) {
        ImGui::Text(("MeshGenerated: " + std::string(m_meshGenerated ? "Yes" : "No")).c_str());
        ImGui::Text(("FoliageGenerated: " + std::string(m_foliageGenerated ? "Yes" : "No")).c_str());
        ImGui::Text(("ActiveLength: " + std::to_string(m_activeLength)).c_str());
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Parameters")) {
        m_parameters.OnGui();
        ImGui::TreePop();
    }
}

void TreeData::ExportModel(const std::string &filename, const bool &includeFoliage) const {
    auto mesh = GetOwner().GetPrivateComponent<MeshRenderer>().m_mesh;
    if (!mesh) return;
    if (mesh->GetVerticesAmount() == 0) {
        Debug::Log("Mesh not generated!");
        return;
    }
    auto triangles = mesh->UnsafeGetTriangles();
    std::ofstream of;
    of.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (of.is_open()) {
        std::string branchVertices;
        std::string branchletVertices;
        std::string leafVertices;
        std::string branchIndices;
        std::string branchletIndices;
        std::string leafIndices;
#pragma region Data collection
        for (const auto &vertex : mesh->UnsafeGetVertices()) {
            branchVertices += "v " + std::to_string(vertex.m_position.x)
                              + " " + std::to_string(vertex.m_position.y)
                              + " " + std::to_string(vertex.m_position.z)
                              + "\n";
        }
        for (int i = 0; i < triangles.size(); i++) {
            const auto triangle = triangles[i];
            branchIndices += "f " + std::to_string(triangle.x + 1)
                             + " " + std::to_string(triangle.y + 1)
                             + " " + std::to_string(triangle.z + 1)
                             + "\n";
        }
#pragma endregion
        size_t branchVerticesSize = mesh->GetVerticesAmount();
        if (includeFoliage) {
            Entity foliageEntity;
            GetOwner().ForEachChild([&foliageEntity](Entity child) {
                                        /*
                                            if (child.HasComponentData<WillowFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else if (child.HasComponentData<AppleFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else if (child.HasComponentData<AcaciaFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else if (child.HasComponentData<BirchFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else if (child.HasComponentData<OakFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else if (child.HasComponentData<MapleFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }
                                            else */if (child.HasDataComponent<TreeLeavesTag>()) {
                                        foliageEntity = child;
                                    }
                                        /*
                                            else if (child.HasComponentData<PineFoliageInfo>())
                                            {
                                                foliageEntity = child;
                                            }*/
                                    }
            );
            size_t branchletVerticesSize = 0;
            if (foliageEntity.HasPrivateComponent<MeshRenderer>()) {
                mesh = foliageEntity.GetPrivateComponent<MeshRenderer>().m_mesh;
                triangles = mesh->UnsafeGetTriangles();
                branchletVerticesSize += mesh->GetVerticesAmount();
#pragma region Data collection
                for (const auto &vertex : mesh->UnsafeGetVertices()) {
                    branchletVertices += "v " + std::to_string(vertex.m_position.x)
                                         + " " + std::to_string(vertex.m_position.y)
                                         + " " + std::to_string(vertex.m_position.z)
                                         + "\n";
                }
                for (auto triangle : triangles) {
                    branchletIndices += "f " + std::to_string(triangle.x + branchVerticesSize + 1)
                                        + " " + std::to_string(triangle.y + branchVerticesSize + 1)
                                        + " " + std::to_string(triangle.z + branchVerticesSize + 1)
                                        + "\n";
                }
#pragma endregion
            }

            if (foliageEntity.HasPrivateComponent<Particles>()) {
                auto &particles = foliageEntity.GetPrivateComponent<Particles>();
                mesh = particles.m_mesh;
                triangles = mesh->UnsafeGetTriangles();
                auto &matrices = particles.m_matrices->m_value;
                size_t offset = 0;
                for (auto &matrix : matrices) {
                    for (const auto &vertex : mesh->UnsafeGetVertices()) {
                        glm::vec3 position = matrix * glm::vec4(vertex.m_position, 1);
                        leafVertices += "v " + std::to_string(position.x)
                                        + " " + std::to_string(position.y)
                                        + " " + std::to_string(position.z)
                                        + "\n";
                    }
                }
                for (auto &matrix : matrices) {
                    for (auto triangle : triangles) {
                        leafIndices += "f " + std::to_string(
                                triangle.x + offset + branchVerticesSize + branchletVerticesSize + 1)
                                       + " " + std::to_string(
                                triangle.y + offset + branchVerticesSize + branchletVerticesSize + 1)
                                       + " " + std::to_string(
                                triangle.z + offset + branchVerticesSize + branchletVerticesSize + 1)
                                       + "\n";
                    }
                    offset += mesh->GetVerticesAmount();
                }
            }
        }

        of.write(branchVertices.c_str(), branchVertices.size());
        of.flush();
        if (!branchletVertices.empty()) {
            of.write(branchletVertices.c_str(), branchletVertices.size());
            of.flush();
        }
        if (!leafVertices.empty()) {
            of.write(leafVertices.c_str(), leafVertices.size());
            of.flush();
        }
        std::string group = "o branches\n";
        of.write(group.c_str(), group.size());
        of.write(branchIndices.c_str(), branchIndices.size());
        of.flush();
        if (!branchletVertices.empty()) {
            group = "o branchlets\n";
            of.write(group.c_str(), group.size());
            of.write(branchletIndices.c_str(), branchletIndices.size());
            of.flush();
        }
        if (!leafVertices.empty()) {
            group = "o leaves\n";
            of.write(group.c_str(), group.size());
            of.write(leafIndices.c_str(), leafIndices.size());
            of.flush();
        }
        of.close();
        Debug::Log("Model saved as " + filename);
    } else {
        Debug::Error("Can't open file!");
    }
}



