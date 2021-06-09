from setuptools import setup

package_name = "top_map"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="justin",
    maintainer_email="jbwasse2@illinois.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "extractor = top_map.extractor:main",
            "similarityServer = top_map.similarityServer:main",
            "similarityClient = top_map.similarityClient:main",
        ],
    },
)
