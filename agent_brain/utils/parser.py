from json_repair import repair_json


def parse_json_response(response: str) -> dict:
    try:
        repaired = repair_json(response, return_objects=True)
        return repaired if isinstance(repaired, dict) else {}
    except Exception as e:
        print(f"Error repairing JSON: {e}")
        return {}
