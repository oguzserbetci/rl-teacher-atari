from setuptools import setup

setup(name='simple_trpo',
    version='0.0.1',
    install_requires=[
        'gym[mujoco]==0.9.1',
        'mujoco-py==0.5.7',
        'tensorflow',
        'cloudpickle'
    ]
)
