#include "Spline.hpp"
#include "SorghumLayer.hpp"

using namespace SorghumFactory;

SplineNode::SplineNode() {}
SplineNode::SplineNode(glm::vec3 position, float angle, float width,
                       float waviness, glm::vec3 axis, bool isLeaf,
                       float surfacePush, float range) {
  m_position = position;
  m_theta = angle;
  m_width = width;
  m_waviness = waviness;
  m_axis = axis;
  m_isLeaf = isLeaf;
  m_surfacePush = surfacePush;
  m_range = range;
}
