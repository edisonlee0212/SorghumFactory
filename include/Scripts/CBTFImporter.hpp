#pragma once
#include <SorghumLayer.hpp>

using namespace EcoSysLab;

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