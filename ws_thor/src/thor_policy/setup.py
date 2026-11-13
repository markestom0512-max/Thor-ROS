from glob import glob

from setuptools import find_packages, setup

package_name = "thor_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/models", glob("models/*.onnx")),
        ("share/" + package_name + "/data", glob("data/*.npz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tom",
    maintainer_email="tom@todo.todo",
    description="Policy de reach Isaac Lab pour Thor (ONNX, 30 Hz).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = thor_policy.policy_node:main",
            "check_golden = thor_policy.check_golden:main",
        ],
    },
)
