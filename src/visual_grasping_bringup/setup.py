from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_grasping_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description=(
        'YOLOv8 grasping package with elongated-object-box grasping '
        'and dynamic collision objects.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_grasping = visual_grasping_bringup.visual_grasping_node:main',
            'dynamic_collision_objects = visual_grasping_bringup.dynamic_collision_objects_node:main',
        ],
    },
)
