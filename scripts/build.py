import os
from hatchling.builders.wheel import WheelBuilder

WHEEL_DIR = "adrs/wheels"


def get_builder():
    return CustomBuilder


def get_tags() -> list[str]:
    return [
        "cp313-cp313-win_amd64",
        "cp314-cp314-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2",
        "cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64",
        "cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64",
        "cp312-cp312-win_amd64",
        "cp312-cp312-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2",
        "cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64",
        "cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64",
        "cp314-cp314-win_amd64",
        "cp314-cp314-manylinux_2_17_aarch64.manylinux2014_aarch64",
        "cp313-cp313-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2",
        "cp313-cp313-manylinux_2_17_aarch64.manylinux2014_aarch64",
    ]


def ensure_wheels_exist() -> str | None:
    deps: list[list[str]] = []
    with open("adrs/installer.py", "r") as f:
        deps = list(
            map(
                lambda line: [
                    s[1:-1]
                    for s in line.rstrip().split("(")[1].split(")")[0].split(", ")
                ],
                filter(lambda line: line.startswith("    ensure_one("), f.readlines()),
            )
        )

    tags = get_tags()
    required = [f"{pkg}-{version}-{tag}.whl" for [pkg, version] in deps for tag in tags]

    exists = os.listdir(WHEEL_DIR)
    for req in required:
        if req not in exists:
            return req


class CustomBuilder(WheelBuilder):
    PLUGIN_NAME = "custom"

    def build_standard(self, directory, **build_data):
        targets = ""

        missing_wheel = ensure_wheels_exist()
        if missing_wheel is not None:
            raise RuntimeError(
                f"Wheel {missing_wheel} is missing from {WHEEL_DIR}, cannot build."
            )

        for tag in get_tags():
            self.build_config["targets"]["custom"]["artifacts"] = [
                f"{WHEEL_DIR}/*{tag}.whl"
            ]
            build_data["tag"] = tag
            target = super().build_standard(directory, **build_data)
            targets += f"dist/{target.split('dist/')[-1]}\n"

        return targets
