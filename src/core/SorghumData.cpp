#include <SorghumData.hpp>
#include <SorghumManager.hpp>
using namespace PlantFactory;
SorghumData::SorghumData()
{
}

SorghumData::~SorghumData()
{
	Entity rootInternode;
	EntityManager::ForEachChild(GetOwner(), [&](Entity child)
		{
			if (child.HasComponentData<InternodeInfo>()) rootInternode = child;
		}
	);
	if (rootInternode.IsValid()) EntityManager::DeleteEntity(rootInternode);
}

void SorghumData::OnGui()
{
	if (ImGui::TreeNodeEx("I/O"))
	{
		if (m_meshGenerated) {
			if (ImGui::Button("Export Model")) {
				auto result = FileIO::SaveFile("3D model (*.obj)\0*.obj\0");
				if (result.has_value())
				{
					const std::string path = result.value();
					if (!path.empty())
					{
						ExportModel(path);
					}
				}
			}
		}
		ImGui::TreePop();
	}
	if (ImGui::TreeNodeEx("Parameters"))
	{
		m_parameters.OnGui();
		ImGui::TreePop();
	}
}

void SorghumData::ExportModel(const std::string& filename, const bool& includeFoliage) const
{
	std::ofstream of;
	of.open(filename, std::ofstream::out | std::ofstream::trunc);
	if (of.is_open())
	{
		std::string start = "#Sorghum field, by Bosheng Li";
		start += "\n";
		of.write(start.c_str(), start.size());
		of.flush();
		unsigned startIndex = 1;
		SorghumManager::ExportSorghum(GetOwner(), of, startIndex);
		of.close();
		Debug::Log("Sorghums saved as " + filename);
	}
	else
	{
		Debug::Error("Can't open file!");
	}
}
