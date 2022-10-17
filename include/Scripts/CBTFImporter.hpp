#pragma once
#include <SorghumLayer.hpp>

using namespace PlantArchitect;

namespace Scripts {
class CBTFImporter : public IPrivateComponent{
public:
  bool m_processing = false;
  std::filesystem::path m_currentExportFolder;
  std::vector<std::filesystem::path> m_importFolders;
  void OnInspect() override;
  void Update() override;
};
}