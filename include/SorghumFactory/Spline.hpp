#pragma once
#include <CUDAModule.hpp>
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include <RayTracedRenderer.hpp>
#include <sorghum_factory_export.h>
using namespace UniEngine;
namespace SorghumFactory {
struct SORGHUM_FACTORY_API PlantNode {
  glm::vec3 m_position;
  float m_theta;
  float m_width;
  glm::vec3 m_axis;
  bool m_isLeaf;
  PlantNode(glm::vec3 position, float angle, float width, glm::vec3 axis,
            bool isLeaf);
};

class SORGHUM_FACTORY_API Spline : public IPrivateComponent {
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
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
};
} // namespace SorghumFactory