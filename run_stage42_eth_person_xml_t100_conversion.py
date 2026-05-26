from src.stage42_eth_person_xml_t100_conversion import run_stage42_eth_person_xml_t100_conversion


if __name__ == "__main__":
    result = run_stage42_eth_person_xml_t100_conversion()
    gate = result["stage42_bl_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
