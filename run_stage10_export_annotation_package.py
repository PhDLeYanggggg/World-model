import json

from src.annotation.stage10_annotation_export import export_annotation_package


if __name__ == "__main__":
    print(json.dumps({"exported_annotations": export_annotation_package()}, indent=2))
